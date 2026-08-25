"""Shared visual-grounding, artifact, and multi-stage routing helpers."""

import io
import json
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw
from google.genai import types
from gemini_meter import metered_generate_content


MARKER_COLOR = (255, 0, 180)


def image_png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def target_bbox_in_crop(full_bbox, crop_box, full_size):
    """Convert a normalized full-image bbox to normalized crop coordinates."""
    full_w, full_h = full_size
    cx1, cy1, cx2, cy2 = crop_box
    crop_w = max(1, cx2 - cx1)
    crop_h = max(1, cy2 - cy1)
    tx1 = (full_bbox[0] * full_w - cx1) / crop_w
    ty1 = (full_bbox[1] * full_h - cy1) / crop_h
    tx2 = (full_bbox[2] * full_w - cx1) / crop_w
    ty2 = (full_bbox[3] * full_h - cy1) / crop_h
    return [
        min(max(tx1, 0.0), 1.0),
        min(max(ty1, 0.0), 1.0),
        min(max(tx2, 0.0), 1.0),
        min(max(ty2, 0.0), 1.0),
    ]


def make_marked_crop(clean_crop, local_bbox):
    """Draw a non-destructive target marker on a copy of the clean crop."""
    marked = clean_crop.copy().convert("RGB")
    width, height = marked.size
    x1 = int(local_bbox[0] * width)
    y1 = int(local_bbox[1] * height)
    x2 = int(local_bbox[2] * width)
    y2 = int(local_bbox[3] * height)
    line_width = max(2, min(width, height) // 80)
    inset = line_width + 1
    x1 = max(0, x1 - inset)
    y1 = max(0, y1 - inset)
    x2 = min(width - 1, x2 + inset)
    y2 = min(height - 1, y2 + inset)
    draw = ImageDraw.Draw(marked)
    draw.rectangle((x1, y1, x2, y2), outline=MARKER_COLOR, width=line_width)
    return marked


def round_artifact_dir(root, fname, round_idx, route="single", stage=None):
    stem = Path(fname).stem
    directory = Path(root) / stem / f"round_{round_idx:02d}" / route
    if stage is not None:
        directory = directory / f"stage_{stage:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_image(directory, name, image):
    path = Path(directory) / name
    image.convert("RGB").save(path)
    return str(path)


def save_json(directory, name, payload):
    path = Path(directory) / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def prepare_round_artifacts(root, fname, round_idx, full_before, clean_crop,
                            crop_box, target_bbox):
    directory = round_artifact_dir(root, fname, round_idx)
    local_bbox = target_bbox_in_crop(target_bbox, crop_box, full_before.size)
    marked_crop = make_marked_crop(clean_crop, local_bbox)
    save_image(directory, "before_full.png", full_before)
    save_image(directory, "clean_crop.png", clean_crop)
    save_image(directory, "initial_marked_crop.png", marked_crop)
    save_json(directory, "initial_target.json", {
        "full_bbox": target_bbox,
        "crop_box_pixels": list(crop_box),
        "target_bbox_in_crop": local_bbox,
    })
    return directory, marked_crop


def plan_multistage_route(client, model, token_log, clean_crop, marked_crop,
                          rejected_crop, edit_summary, verifier_reason, retries=2):
    """Escalate a rejected single edit only when decomposition is visually useful."""
    prompt = f"""Inspect these three crop images:
IMAGE 1 is the clean crop before editing.
IMAGE 2 marks the intended target with a magenta rectangle.
IMAGE 3 is the rejected single-edit result.

Requested correction:
{edit_summary}

Verifier reason:
{verifier_reason}

Choose a recovery route.
- retry_single: the task is atomic and should be attempted again with a better short instruction.
- relocalize: the marked target or required destination is absent/wrong.
- multi_stage: the edit genuinely requires clearing/rebuilding/reflowing content in 2 or 3 dependent stages.

Use multi_stage for structural cases such as removing an incomplete legend before rebuilding it, clearing insufficient space before adding complete content, or replacing a complex grouped component. Do not use it merely because one stochastic edit failed.

For every stage, describe only one crop-local action. Intermediate stages may temporarily remove content. Do not mention bbox or full-image coordinates.
Return STRICT JSON only:
{{
  "route": "retry_single | relocalize | multi_stage",
  "reason": "short reason",
  "stages": [
    {{"operation": "remove | add | replace | reflow | refine", "instruction": "short action", "expected_state": "visible state required after this stage"}}
  ]
}}"""
    for attempt in range(retries):
        try:
            response = metered_generate_content(
                client, token_log, model=model,
                contents=[
                    types.Part(text=prompt),
                    types.Part(text="IMAGE 1 — CLEAN CROP:"),
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=image_png_bytes(clean_crop))),
                    types.Part(text="IMAGE 2 — MARKED TARGET:"),
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=image_png_bytes(marked_crop))),
                    types.Part(text="IMAGE 3 — REJECTED RESULT:"),
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=image_png_bytes(rejected_crop))),
                ],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            raw = response.text.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("no JSON route object")
            route = json.loads(match.group())
            choice = route.get("route")
            stages = route.get("stages") or []
            if choice not in {"retry_single", "relocalize", "multi_stage"}:
                raise ValueError("invalid route")
            if choice == "multi_stage" and not (2 <= len(stages) <= 3):
                raise ValueError("multi_stage must contain 2 or 3 stages")
            route["raw_response"] = raw
            return route
        except Exception as ex:
            if attempt + 1 == retries:
                return {"route": "retry_single", "reason": str(ex), "stages": []}
    return {"route": "retry_single", "reason": "routing unavailable", "stages": []}


def verify_intermediate_stage(client, model, before_crop, after_crop, stage, retries=2):
    """Verify an intermediate goal without requiring the final edit to be complete."""
    prompt = f"""Verify one intermediate image-editing stage.

Stage instruction: {stage['instruction']}
Expected visible state after this stage: {stage['expected_state']}

Compare BEFORE and AFTER. Judge only this intermediate stage, not the final overall task.
Return STRICT JSON only:
{{"stage_achieved": true, "collateral_damage": false, "confidence": 0.0, "reason": "short reason"}}

Reject unintended changes, duplicated content, broken text, broken lines, artifacts, or damage outside the stage target."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part(text=prompt + "\n\nBEFORE:"),
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=image_png_bytes(before_crop))),
                    types.Part(text="AFTER:"),
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=image_png_bytes(after_crop))),
                ],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            raw = response.text.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("no JSON stage verification object")
            result = json.loads(match.group())
            confidence = float(result.get("confidence", 0.0))
            accepted = (
                result.get("stage_achieved") is True
                and result.get("collateral_damage") is False
                and confidence >= 0.65
            )
            result.update({"accepted": accepted, "raw_response": raw})
            return accepted, result
        except Exception as ex:
            if attempt + 1 == retries:
                return False, {"accepted": False, "reason": str(ex)}
    return False, {"accepted": False, "reason": "verification unavailable"}


def execute_multistage(clean_crop, marked_crop, stages, round_directory, seed_base,
                       edit_callback, verify_callback):
    """Run a transactional crop rebuild; return the original crop on any failure."""
    stage_work = clean_crop.copy()
    records = []
    for index, stage in enumerate(stages, start=1):
        directory = Path(round_directory) / "multistage" / f"stage_{index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        save_image(directory, "before.png", stage_work)
        save_image(directory, "marked_reference.png", marked_crop)
        instruction = stage["instruction"].strip()
        prefix = "Edit image 1 only; image 2 marks the target."
        if not instruction.lower().startswith(prefix.lower()):
            instruction = f"{prefix} {instruction}"
        if not instruction.endswith("Keep everything else unchanged."):
            instruction += " Keep everything else unchanged."
        output = edit_callback(stage_work, marked_crop, instruction, seed_base + index)
        output = output.convert("RGB").resize(stage_work.size)
        save_image(directory, "qwen_output.png", output)
        accepted, verification = verify_callback(stage_work, output, stage)
        save_json(directory, "verification.json", verification)
        records.append({
            "stage": index,
            "instruction": instruction,
            "accepted": accepted,
            "verification": verification,
        })
        if not accepted:
            return False, clean_crop.copy(), records
        stage_work = output
    return True, stage_work, records


def recover_with_multistage(client, model, token_log, clean_crop, marked_crop,
                            rejected_crop, edit_summary, verifier_reason,
                            round_directory, seed_base, edit_callback):
    route = plan_multistage_route(
        client, model, token_log, clean_crop, marked_crop, rejected_crop,
        edit_summary, verifier_reason,
    )
    save_json(round_directory, "route_decision.json", route)
    if route.get("route") != "multi_stage":
        return route, False, clean_crop.copy(), []
    success, output, records = execute_multistage(
        clean_crop, marked_crop, route["stages"], round_directory, seed_base,
        edit_callback=edit_callback,
        verify_callback=lambda before, after, stage: verify_intermediate_stage(
            client, model, before, after, stage
        ),
    )
    return route, success, output, records
