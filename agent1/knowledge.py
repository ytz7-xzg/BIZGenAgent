#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boogu-Turbo knowledge 维度修复流水线（与参考脚本 bizgen_knowledge_fix.py 逻辑一致）
流程：mark/ 评测 json → 找 knowledge 维度 result=false 的题
     → Gemini 看图判断知识错误 + 给出局部编辑计划
     → Qwen-Image-Edit-2511 裁区域局部修图 → 羽化贴回
     → 存 8-19edit/（文件名与原图一致，可直接重评）

用法（bizgeneval 环境，2 号卡）：
    conda activate bizgeneval
    CUDA_VISIBLE_DEVICES=2 python boogu_knowledge_fix.py
"""

import os, io, json, re
from datetime import datetime
import numpy as np
from PIL import Image, ImageFilter
import torch
if os.environ.get("BIZ_EDIT_BACKEND", "qwen").lower() == "sensenova":
    from repair_visual_utils import SenseNovaEditPipeline as QwenImageEditPlusPipeline
else:
    from diffusers import QwenImageEditPlusPipeline
from google import genai
from google.genai import types
from gemini_meter import metered_generate_content
from repair_visual_utils import make_marked_crop, prepare_round_artifacts, recover_with_multistage, save_image, save_json

# ================== 配置（要改就改这里） ==================
LIMIT = None
SEED = 42
SKIP_EXISTING = True
TARGET_DIMENSION = "knowledge"
VISUAL_ANCHOR_PROMPT_V2 = True
MAX_KNOWLEDGE_ROUNDS = 10
MAX_REJECTIONS_PER_ISSUE = 3

DATA_PATH = "/mmu-vcg/zb08/zixuan/BIZ/results/sample100/sample100.jsonl"
IMG_DIR = "/mmu-vcg/zb08/zixuan/BIZ/results/sample100/originals"
MARK_DIR = "/mmu-vcg/zb08/zixuan/BIZ/results/sample100/mark_original"
OUTPUT_DIR = os.environ.get("BIZ_OUTPUT_DIR", "/mmu-vcg/zb08/zixuan/BIZ/results/agent1_repair/edited")
PLAN_DIR = os.environ.get("BIZ_PLAN_DIR", "/mmu-vcg/zb08/zixuan/BIZ/results/agent1_repair/plans/knowledge")
INTERMEDIATE_DIR = os.path.join(PLAN_DIR, "intermediates")

EDIT_PIPE_PATH = os.environ.get("BIZ_EDIT_MODEL_PATH", "/mmu-vcg/zb08/CKPTS/qwen-edit_2511")

KEY_PATH = "/mmu-vcg/zb08/llm-6669-1b56d4a3712d.json"
GEMINI_MODEL = "gemini-3-flash-preview"

REGION_PAD = 0.15       # bbox 自身宽高的外扩比例
MAX_REGION_PAD = 0.05   # 单侧最多外扩全图 5%
MIN_REGION_PAD_PX = 24  # 单侧至少保留 24px 上下文
EDIT_STEPS = 40
FEATHER = 24
VERIFY_CONFIDENCE = 0.65             # 贴回时的羽化像素
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLAN_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

# Gemini 客户端：优先用 vertex 服务账号（和参考脚本一致），否则走 GEMINI_API_KEY
if os.path.exists(KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
    client = genai.Client(vertexai=True, project="llm-6669", location="global")
    print("[INFO] Gemini 使用 vertex 服务账号")
else:
    client = genai.Client()
    print("[INFO] Gemini 使用 GEMINI_API_KEY")


def extract_image(output, depth=0):
    if depth > 5:
        return output
    if hasattr(output, "save"):
        return output
    if hasattr(output, "images"):
        imgs = output.images
        if isinstance(imgs, (list, tuple)) and len(imgs) > 0:
            return extract_image(imgs[0], depth + 1)
    if isinstance(output, (list, tuple)) and len(output) > 0:
        return extract_image(output[0], depth + 1)
    return output


def image_to_bytes(image_source):
    """支持路径或 PIL Image；多轮规划时始终发送最新工作图。"""
    if isinstance(image_source, (str, os.PathLike)):
        with open(image_source, "rb") as f:
            return f.read()
    buf = io.BytesIO()
    image_source.save(buf, format="PNG")
    return buf.getvalue()


def normalize_bbox(raw_bbox):
    """统一成 0~1 的 xyxy；兼容 Gemini 偶尔返回 0~1000 或 xywh。"""
    if len(raw_bbox) != 4:
        return None
    try:
        x1, y1, a, b = [float(v) for v in raw_bbox]
    except Exception:
        return None

    if max(x1, y1, a, b) > 1.5:
        x1, y1, a, b = [v / 1000.0 for v in (x1, y1, a, b)]

    if a <= x1 or b <= y1:
        x2, y2 = x1 + a, y1 + b
    else:
        x2, y2 = a, b

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    bb = [
        min(max(x1, 0.0), 1.0),
        min(max(y1, 0.0), 1.0),
        min(max(x2, 0.0), 1.0),
        min(max(y2, 0.0), 1.0),
    ]
    if bb[2] - bb[0] < 0.01 or bb[3] - bb[1] < 0.01:
        return None
    return bb


def clean_field(value, default=""):
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def build_instruction(e):
    instruction = clean_field(e.get("instruction"))
    if instruction:
        return instruction
    target = clean_field(e.get("target_object"), "target content")
    correct = clean_field(e.get("correct_content"))
    if correct:
        return f"Replace only the {target} with the correct content: {correct}."
    return ""



def normalize_edit(e):
    if not isinstance(e, dict):
        return None
    bb = normalize_bbox(e.get("bbox", []))
    instruction = build_instruction(e)
    if bb is None or not instruction:
        return None
    target_object = clean_field(e.get("target_object"), "target content")
    return {
        "knowledge_type": clean_field(e.get("knowledge_type"), "other"),
        "target_object": target_object,
        "failed_check": clean_field(e.get("failed_check"), target_object),
        "visual_signature": clean_field(e.get("visual_signature"), target_object),
        "reference_context": clean_field(e.get("reference_context")),
        "preserve": clean_field(
            e.get("preserve"),
            "the original layout, style, position, and all unrelated content",
        ),
        "current_content": clean_field(e.get("current_content"), "current incorrect content"),
        "correct_content": clean_field(e.get("correct_content")),
        "correction_reason": clean_field(e.get("correction_reason"), "The visible content does not match the original prompt."),
        "bbox": bb,
        "instruction": instruction,
    }



def compose_qwen_prompt(e):
    """Deterministic fallback when crop-level Gemini prompting is unavailable."""
    return (
        f"Replace only the visible content '{e['current_content']}' with exactly '{e['correct_content']}'. "
        "Keep everything else unchanged."
    )


def gemini_crop_instruction(region_img, marked_img, edit, retries=3):
    """Let Gemini inspect only the crop and write the instruction Qwen will see."""
    buf = io.BytesIO()
    marked_img.save(buf, format="PNG")
    prompt = f"""Inspect only the attached cropped image.

The incorrect visible content is: {edit['current_content']}
The exact required content is: {edit['correct_content']}

The magenta rectangle marks the target. Write one very short instruction for editing IMAGE 1; IMAGE 2 will be this marked reference.
Rules:
- Begin with: Edit image 1 only; image 2 marks the target.
- Identify the marked target with exact visible content plus at most one appearance cue.
- State exactly one replacement action.
- Do not mention bbox, coordinates, row/column numbers, ordering in the full image, or anything outside this crop.
- Do not include analysis or grounding explanations.
- End with: Keep everything else unchanged.
- Refine the target rectangle in crop-local normalized xyxy coordinates.
- Return STRICT JSON only: {{"instruction": "...", "target_bbox": [x1, y1, x2, y2]}}"""

    for attempt in range(retries):
        try:
            resp = metered_generate_content(
                client, os.environ['GEMINI_TOKEN_LOG'], model=GEMINI_MODEL,
                contents=[
                    types.Part(text=prompt),
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=buf.getvalue())),
                ],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            raw = resp.text.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("no JSON object in crop-instruction response")
            payload = json.loads(match.group())
            instruction = clean_field(payload.get("instruction"))
            refined_bbox = normalize_bbox(payload.get("target_bbox", []))
            if not instruction:
                raise ValueError("empty crop instruction")
            prefix = "Edit image 1 only; image 2 marks the target."
            if not instruction.lower().startswith(prefix.lower()):
                instruction = f"{prefix} {instruction}"
            return instruction, refined_bbox
        except Exception as ex:
            print(f"  [CROP PROMPT retry {attempt+1}/{retries}] {str(ex)[:120]}")
    return "Edit image 1 only; image 2 marks the target. " + compose_qwen_prompt(edit), None


def gemini_plan(image_source, prompt, failed_questions, applied_history=None,
                blocked_checks=None, rejection_history=None, retries=3):
    """Gemini 看最新工作图，自主发现下一处知识问题。"""
    applied_history = applied_history or []
    blocked_checks = blocked_checks or set()
    rejection_history = rejection_history or []
    history_text = "\n".join(f"- {x}" for x in applied_history) or "None."
    blocked_text = "\n".join(f"- {x}" for x in sorted(blocked_checks)) or "None."
    rejection_text = "\n".join(f"- {x}" for x in rejection_history[-6:]) or "None."
    planner_prompt = f"""This image was generated from the prompt:
\"\"\"{prompt}\"\"\"

Edits already accepted in previous rounds:
{history_text}

Issues blocked after repeated rejected candidates:
{blocked_text}

Recent rejected candidates and verifier reasons:
{rejection_text}

Independently inspect the image against the original generation prompt.
Discover visible factual or knowledge discrepancies yourself; do not assume an evaluator has identified them for you.
Plan minimal local edits only for discrepancies visibly supported by the image and original prompt.
Return STRICT JSON (no markdown, no explanation), a list:
[{{
  "knowledge_type": "fact | number | formula | date | name | label | other",
  "target_object": "exact visual element",
  "failed_check": "a concise description of the self-discovered visible discrepancy",
  "visual_signature": "current visible content plus its style and nearby local cues",
  "reference_context": "nearby diagram, axis, heading, row, or label needed to interpret the correction; empty if unnecessary",
  "preserve": "layout, style, position, and unrelated content to keep unchanged",
  "current_content": "currently visible incorrect content",
  "correct_content": "exact required final content",
  "correction_reason": "why the replacement is factually correct",
  "bbox": [x1, y1, x2, y2],
  "instruction": "one exact correction command"
}}]

VISUAL GROUNDING RULES:
- failed_check MUST describe one knowledge discrepancy independently discovered from the image and original prompt. Never select an issue shown in the blocked list.
- visual_signature MUST identify the target by how it looks NOW inside the crop. Use 2-4 observable cues such as object type, current color/shape, unique attached text or number, and a nearby local relationship.
- Do NOT rely only on a semantic category name, legend label, series name, or an ordinal such as "third from the left". Those labels or orders may already be wrong.
- instruction MUST be one atomic action on the visually identified target. Use "it" or "its" instead of repeating an unreliable semantic name.
- Do not combine independent actions. For example, changing a bar color and changing its height are two rounds. Return the single safest action first; the next round will inspect the updated image again.
- reference_context names the visible guide needed to perform the action. For a measured position or size, name the exact axis tick, gridline, edge, or neighboring object.
- bbox MUST include the complete target, all cues named in visual_signature, and the visible reference_context. Everything needed by Qwen must be visible in this one crop.
- preserve MUST explicitly name the target properties and nearby content that must stay unchanged, excluding only the property changed by this action.

Rules:
- bbox: normalized [0,1] coordinates tightly around the visual element whose knowledge content is wrong or missing.
- instruction: a short direct command stating EXACTLY what the corrected content should be,
  e.g. "Change the formula to '2H2O2 -> 2H2O + O2'" or "Replace the label with 'manganese dioxide'".
- Put the single safest edit FIRST. The pipeline applies only that edit and then replans from the updated image.
- If no visible knowledge discrepancy remains, return an empty list: [].
- One atomic action per round. Never merge color, height, size, text, or position changes.
- Knowledge corrections only, no layout changes."""

    img_bytes = image_to_bytes(image_source)

    attempts = []
    for attempt in range(retries):
        try:
            resp = metered_generate_content(client, os.environ['GEMINI_TOKEN_LOG'],
                model=GEMINI_MODEL,
                contents=[
                    types.Part(text=planner_prompt),
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=img_bytes)),
                ],
                config=types.GenerateContentConfig(temperature=0.1),
            )
            text = resp.text.strip()
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                raise ValueError("no JSON list in response")
            payload = json.loads(match.group())
            if isinstance(payload, dict):
                payload = payload.get("edits", [])
            if not isinstance(payload, list):
                raise ValueError("Gemini response is not an edit list")

            if len(payload) == 0:
                return [], {
                    "status": "no_remaining_issues",
                    "planner_model": GEMINI_MODEL,
                    "planner_prompt": planner_prompt,
                    "raw_response": text,
                    "parsed_edits": [],
                    "attempts": attempts + [{
                        "attempt": attempt + 1,
                        "status": "no_remaining_issues",
                        "raw_response": text,
                    }],
                }

            valid = []
            for e in payload:
                item = normalize_edit(e)
                if item is not None:
                    valid.append(item)

            attempts.append({
                "attempt": attempt + 1,
                "status": "success" if valid else "no_valid_edits",
                "raw_response": text,
            })

            if not valid:
                print(f"  [RAW] {text[:300]}")
                raise ValueError("no valid edits parsed")

            return valid, {
                "status": "success",
                "planner_model": GEMINI_MODEL,
                "planner_prompt": planner_prompt,
                "raw_response": text,
                "parsed_edits": valid,
                "attempts": attempts,
            }
        except Exception as ex:
            print(f"  [GEMINI retry {attempt+1}/{retries}] {str(ex)[:120]}")
            if not attempts or attempts[-1].get("attempt") != attempt + 1:
                attempts.append({
                    "attempt": attempt + 1,
                    "status": "error",
                    "error": str(ex),
                })

    return None, {
        "status": "failed",
        "planner_model": GEMINI_MODEL,
        "planner_prompt": planner_prompt,
        "parsed_edits": [],
        "attempts": attempts,
    }


def save_plan_json(fname, it, img_path, save_path, failed, plan_info,
                   round_idx, selected_edit, planned_count, applied_history):
    plan_path = os.path.join(
        PLAN_DIR, fname.replace(".png", f"_round{round_idx}.json")
    )
    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "file_name": fname,
        "domain": it.get("domain"),
        "dimension": it.get("dimension"),
        "id": it.get("id"),
        "source_image": img_path,
        "output_image": save_path,
        "original_prompt": it.get("prompt", ""),
        "failed_questions": failed,
        "round": round_idx,
        "selected_edit": selected_edit,
        "planned_edit_count": planned_count,
        "applied_history_before_round": applied_history,
        **plan_info,
    }
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return plan_path


def crop_region(img, bbox, pad=REGION_PAD):
    W, H = img.size
    # 按 bbox 尺寸动态外扩，避免旧版固定扩全图 15% 造成大范围误改。
    bbox_w = max(0.0, bbox[2] - bbox[0])
    bbox_h = max(0.0, bbox[3] - bbox[1])
    pad_x = min(MAX_REGION_PAD, max(MIN_REGION_PAD_PX / max(W, 1), bbox_w * pad))
    pad_y = min(MAX_REGION_PAD, max(MIN_REGION_PAD_PX / max(H, 1), bbox_h * pad))
    x1 = max(0.0, bbox[0] - pad_x); y1 = max(0.0, bbox[1] - pad_y)
    x2 = min(1.0, bbox[2] + pad_x); y2 = min(1.0, bbox[3] + pad_y)
    box = (int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))
    return img.crop(box), box


def paste_feathered(bg, fg, box, feather=FEATHER):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    fg = fg.resize((w, h))
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)).astype(np.float32)
    alpha = np.clip(d / feather, 0, 1) * 255
    mask = Image.fromarray(alpha.astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(feather // 2))
    bg.paste(fg, (x1, y1), mask)
    return bg


def edit_region(pipe, region_img, qwen_prompt, seed, marked_img=None):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    inputs = {
        "image": [region_img, marked_img] if marked_img is not None else [region_img],
        "prompt": qwen_prompt,
        "generator": generator,
        "true_cfg_scale": 4.0,
        "negative_prompt": "wrong fact, wrong text, unrelated changes, duplicated elements, missing content, distortion, visible seams, redesigned image",
        "num_inference_steps": EDIT_STEPS,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        try:
            output = pipe(**inputs)
        except (TypeError, ValueError) as ex:
            if marked_img is None or not any(word in str(ex).lower() for word in ("image", "list", "input")):
                raise
            inputs["image"] = [region_img]
            inputs["prompt"] = qwen_prompt.replace(
                "Edit image 1 only; image 2 marks the target.",
                "Edit only the precisely described target.",
            )
            output = pipe(**inputs)
    return extract_image(output)


def _pil_png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def verify_candidate(before_img, after_img, original_prompt, edit, failed_questions, retries=2):
    """比较编辑前后；只检查目标是否改善以及是否误伤无关内容。"""
    target_instruction = edit.get("qwen_prompt", edit.get("instruction", ""))
    verifier_prompt = f"""You are checking one local edit in a business image.

Original generation prompt:
\"\"\"{original_prompt}\"\"\"

The attempted local edit was:
\"\"\"{target_instruction}\"\"\"

Compare IMAGE BEFORE and IMAGE AFTER. Judge only visible evidence.
Return STRICT JSON only:
{{
  "target_improved": true,
  "collateral_damage": false,
  "confidence": 0.0,
  "reason": "short visual reason"
}}

Rules:
- target_improved=true only if AFTER visibly moves the target toward the requested correction.
- collateral_damage=true if unrelated text, objects, layout, colors, borders, background, or legibility became worse.
- Reject garbled text, duplicated objects, missing content, obvious seams, or unintended redesign.
- Do not demand perfection; accept a clear local improvement with no obvious collateral damage."""

    attempts = []
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part(text=verifier_prompt + "\n\nIMAGE BEFORE:"),
                    types.Part(inline_data=types.Blob(
                        mime_type="image/png", data=_pil_png_bytes(before_img))),
                    types.Part(text="IMAGE AFTER:"),
                    types.Part(inline_data=types.Blob(
                        mime_type="image/png", data=_pil_png_bytes(after_img))),
                ],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            raw = resp.text.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("no JSON object in verification response")
            result = json.loads(match.group())
            confidence = float(result.get("confidence", 0.0))
            accepted = (
                result.get("target_improved") is True
                and result.get("collateral_damage") is False
                and confidence >= VERIFY_CONFIDENCE
            )
            result.update({
                "status": "accepted" if accepted else "rejected",
                "accepted": accepted,
                "raw_response": raw,
                "attempt": attempt + 1,
            })
            return accepted, result
        except Exception as ex:
            attempts.append({"attempt": attempt + 1, "error": str(ex)})
            print(f"  [VERIFY retry {attempt+1}/{retries}] {str(ex)[:120]}")

    # 验证服务偶发失败时保留原流程结果，避免因 API 故障丢失全部输出。
    return True, {
        "status": "verification_failed_open",
        "accepted": True,
        "reason": "verification unavailable; candidate kept",
        "attempts": attempts,
    }


def save_verification_json(fname, index, info):
    verify_path = os.path.join(
        PLAN_DIR, fname.replace(".png", f"_verify{index}.json")
    )
    with open(verify_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return verify_path


def try_multistage_recovery(pipe, before_full, clean_crop, marked_crop,
                            rejected_crop, crop_box, edit, failed_questions,
                            original_prompt, round_directory, seed_base):
    """Try a transactional multi-stage route after a rejected single edit."""
    route, stages_ok, rebuilt_crop, stage_records = recover_with_multistage(
        client=client,
        model=GEMINI_MODEL,
        token_log=os.environ["GEMINI_TOKEN_LOG"],
        clean_crop=clean_crop,
        marked_crop=marked_crop,
        rejected_crop=rejected_crop,
        edit_summary=edit.get("qwen_prompt", edit.get("instruction", "")),
        verifier_reason=edit.get("_last_rejection_reason", ""),
        round_directory=round_directory,
        seed_base=seed_base,
        edit_callback=lambda work, marker, instruction, seed: edit_region(
            pipe, work, instruction, seed, marked_img=marker
        ),
    )
    if not stages_ok:
        return False, None, {
            "route": route,
            "stages": stage_records,
            "accepted": False,
        }

    rebuilt_full = paste_feathered(
        before_full.copy(), rebuilt_crop.convert("RGB"), crop_box
    )
    save_image(round_directory, "multistage_candidate_full.png", rebuilt_full)
    routed_edit = dict(edit)
    routed_edit["qwen_prompt"] = "Multi-stage route: " + " ".join(
        stage["instruction"] for stage in route["stages"]
    )
    accepted, verification = verify_candidate(
        before_full, rebuilt_full, original_prompt, routed_edit, failed_questions
    )
    result = {
        "route": route,
        "stages": stage_records,
        "final_verification": verification,
        "accepted": accepted,
    }
    save_json(round_directory, "multistage_result.json", result)
    return accepted, rebuilt_full if accepted else None, result



def main():
    # 1. 读数据集，只留 knowledge 维度
    items = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                it = json.loads(line)
                if it.get("dimension") == TARGET_DIMENSION:
                    items.append(it)
    print(f"[INFO] {TARGET_DIMENSION} 维度共 {len(items)} 条")

    # 2. 筛出评测过且有 false 的图（与参考脚本逻辑一致）
    todo = []
    for it in items:
        fname = f"{it['domain']}_{it['dimension']}_{it['id']}.png"
        eval_path = os.path.join(MARK_DIR, fname.replace(".png", ".json"))
        img_path = os.path.join(IMG_DIR, fname)
        if not os.path.exists(eval_path) or not os.path.exists(img_path):
            continue
        with open(eval_path) as f:
            ev = json.load(f)
        failed = [v["raw_description"] for v in ev.get("meta_info", {}).values()
                  if v.get("result") is False and v.get("raw_description")]
        # 同一根问题可能被多道题重复描述，先做精确文本去重。
        failed = list(dict.fromkeys(failed))
        if failed:
            todo.append((it, fname, img_path, failed))

    print(f"[INFO] 有 {TARGET_DIMENSION} 错误的图 {len(todo)} 张，全部处理")
    if LIMIT:
        todo = todo[:LIMIT]
    if not todo:
        print("[INFO] 没有需要处理的图，退出")
        return

    # 3. 加载编辑模型
    print("[LOAD] Loading Qwen-Image-Edit-2511 ...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        EDIT_PIPE_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")
    print("[LOAD] done.")

    # 4. 逐图处理
    n_done, n_skip, n_fail = 0, 0, 0
    for i, (it, fname, img_path, failed) in enumerate(todo):
        save_path = os.path.join(OUTPUT_DIR, fname)
        if SKIP_EXISTING and os.path.exists(save_path):
            print(f"[{i+1}/{len(todo)}] SKIP {fname} (已存在)")
            n_skip += 1
            continue

        print(f"[{i+1}/{len(todo)}] {fname}  失败题数={len(failed)}")
        img = Image.open(img_path).convert("RGB")
        accepted_edits = 0
        applied_history = []
        rejection_counts = {}
        blocked_checks = set()
        rejection_history = []
        try:
            for round_idx in range(1, MAX_KNOWLEDGE_ROUNDS + 1):
                print(f"  [ROUND {round_idx}/{MAX_KNOWLEDGE_ROUNDS}] Gemini 基于当前图重新规划")
                edits, plan_info = gemini_plan(
                    img, it.get("prompt", ""), failed,
                    applied_history=applied_history,
                    blocked_checks=blocked_checks,
                    rejection_history=rejection_history,
                )
                selected_edit = edits[0] if edits else None
                plan_path = save_plan_json(
                    fname, it, img_path, save_path, failed, plan_info,
                    round_idx, selected_edit, len(edits) if edits else 0,
                    list(applied_history),
                )
                print(f"  [PLAN JSON] {plan_path}")
                if edits is None:
                    print("  [STOP] Gemini 请求失败")
                    break
                if len(edits) == 0:
                    print("  [STOP] Gemini 判断当前图没有剩余 knowledge 问题")
                    break

                e = edits[0]
                issue_key = f"{e['knowledge_type']} | {e['current_content']} -> {e['correct_content']}"
                if issue_key in blocked_checks:
                    print(f"  [SKIP ISSUE] 已连续拒绝 {MAX_REJECTIONS_PER_ISSUE} 次：{issue_key[:120]}")
                    continue
                region, box = crop_region(img, e["bbox"])
                if region.size[0] < 32 or region.size[1] < 32:
                    print("  [STOP] 区域太小")
                    break
                before = img.copy()
                round_dir, marked_region = prepare_round_artifacts(
                    INTERMEDIATE_DIR, fname, round_idx, before, region, box, e["bbox"]
                )
                e["qwen_prompt"], refined_bbox = gemini_crop_instruction(region, marked_region, e)
                if refined_bbox is not None:
                    marked_region = make_marked_crop(region, refined_bbox)
                save_image(round_dir, "marked_crop.png", marked_region)
                save_json(round_dir, "refined_target.json", {"target_bbox_in_crop": refined_bbox})
                print(f"  [QWEN PROMPT] {e['qwen_prompt']}")
                save_plan_json(
                    fname, it, img_path, save_path, failed, plan_info,
                    round_idx, e, len(edits), list(applied_history),
                )
                fixed = edit_region(
                    pipe, region, e["qwen_prompt"], SEED + i * 100 + round_idx,
                    marked_img=marked_region,
                )
                save_image(round_dir, "qwen_output_crop.png", fixed)
                candidate = paste_feathered(before.copy(), fixed.convert("RGB"), box)
                save_image(round_dir, "candidate_full.png", candidate)
                accepted, verify_info = verify_candidate(
                    before, candidate, it.get("prompt", ""), e, failed
                )
                verify_info["qwen_prompt"] = e["qwen_prompt"]
                save_json(round_dir, "single_verification.json", verify_info)
                verify_path = save_verification_json(fname, round_idx, verify_info)
                print(f"  [VERIFY] {verify_info['status']} | {verify_path}")
                if accepted:
                    img = candidate
                    accepted_edits += 1
                    applied_history.append(
                        f"Round {round_idx}: {e['current_content']} -> {e['correct_content']}"
                    )
                    rejection_counts.pop(issue_key, None)
                    save_image(round_dir, "committed_full.png", img)
                    save_json(round_dir, "round_status.json", {"status": "accepted_single"})
                    continue

                reason = str(verify_info.get("reason", "no verifier reason"))
                e["_last_rejection_reason"] = reason
                multi_accepted, multi_candidate, multi_info = try_multistage_recovery(
                    pipe, before, region, marked_region, fixed.convert("RGB"), box,
                    e, failed, it.get("prompt", ""), round_dir,
                    SEED + i * 1000 + round_idx * 10,
                )
                if multi_accepted:
                    img = multi_candidate
                    accepted_edits += 1
                    applied_history.append(
                        f"Round {round_idx}: multi-stage {e['current_content']} -> {e['correct_content']}"
                    )
                    rejection_counts.pop(issue_key, None)
                    save_image(round_dir, "committed_full.png", img)
                    save_json(round_dir, "round_status.json", {"status": "accepted_multistage"})
                    continue
                save_json(round_dir, "round_status.json", {
                    "status": "rejected", "reason": reason,
                    "route": multi_info.get("route", {}),
                })
                save_image(round_dir, "rollback_full.png", before)
                rejection_counts[issue_key] = rejection_counts.get(issue_key, 0) + 1
                reject_count = rejection_counts[issue_key]
                rejection_history.append(
                    f"{issue_key} | attempt={reject_count} | reason={reason}"
                )
                print(f"  [ROLLBACK-LOCAL] 保留上一成功版本 ({reject_count}/{MAX_REJECTIONS_PER_ISSUE})")
                if reject_count >= MAX_REJECTIONS_PER_ISSUE:
                    blocked_checks.add(issue_key)
                    print(f"  [SKIP ISSUE] 本问题已屏蔽：{issue_key[:120]}")

            if accepted_edits > 0:
                img.save(save_path)
                print(f"  [SAVE] {save_path}（接受 {accepted_edits} 处编辑）")
                n_done += 1
            else:
                print("  [FAIL] 所有候选编辑均被回退，未保存输出")
                n_fail += 1
        except Exception as ex:
            print(f"  [ERROR] {str(ex)[:200]}")
            if img is not None and accepted_edits > 0:
                img.save(save_path)
                print(f"  [SAVE-PARTIAL] {save_path}（保留此前接受的 {accepted_edits} 处编辑）")
                n_done += 1
            else:
                n_fail += 1

    print(f"\n[DONE] 成功 {n_done} / 跳过 {n_skip} / 失败 {n_fail}")
    print(f"[INFO] Gemini 计划 JSON 保存在：{PLAN_DIR}")


if __name__ == "__main__":
    main()
