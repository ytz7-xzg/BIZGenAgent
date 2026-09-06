#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boogu-Turbo layout 维度修复流水线（迭代版，改编自 bizgen_layout_fix_8_10.py）

核心逻辑：
1. 每张图每轮只执行一个 layout 编辑（优先级最高的那处）。
2. 执行后让 Gemini 基于新图重新规划，避免按旧图 bbox 连续编辑导致区域失效。
3. 每张图最多 5 轮。
4. 移动类编辑支持 source_bbox / target_bbox，裁剪时取并集。
5. bbox 覆盖超过全图 90% 时改为整图编辑，减少误伤。

流程：
mark/ 评测 json → 找 layout 维度 result=false 的题
→ Gemini 看当前图，返回按优先级排序的 layout 编辑计划
→ 只执行第 1 个编辑 → 用编辑后的图重新请求 Gemini
→ 最多重复 MAX_LAYOUT_ROUNDS 轮
→ 结果保存到 8-19edit/（文件名含 domain_layout_id，与其他维度不冲突）

用法（bizgeneval 环境，0 号卡）：
    conda activate bizgeneval
    CUDA_VISIBLE_DEVICES=0 python boogu_layout_fix.py
"""

import os, io, json, re
from datetime import datetime
from PIL import Image
from google import genai
from google.genai import types
from gemini_meter import metered_generate_content
from editor_backend import create_editor

# ================== 配置（要改就改这里） ==================
LIMIT = None
SEED = 42
SKIP_EXISTING = True
TARGET_DIMENSION = "layout"
CLEAN_CROP_PROMPT_V3 = True
MAX_LAYOUT_ROUNDS = 25

DATA_PATH = "/mmu-vcg/zb08/zixuan/BIZ/results/sample100/sample100.jsonl"
IMG_DIR = "/mmu-vcg/zb08/zixuan/BIZ/results/sample100/originals"
MARK_DIR = "/mmu-vcg/zb08/zixuan/BIZ/results/sample100/mark_original"
OUTPUT_DIR = os.environ.get("BIZ_OUTPUT_DIR", "/mmu-vcg/zb08/zixuan/BIZ/results/agent1_repair/edited")
PLAN_DIR = os.environ.get("BIZ_PLAN_DIR", "/mmu-vcg/zb08/zixuan/BIZ/results/agent1_repair/plans/layout")

KEY_PATH = "/mmu-vcg/zb08/llm-6669-1b56d4a3712d.json"
GEMINI_MODEL = os.environ.get("BIZ_GEMINI_MODEL", "gemini-3.8-flash")

REGION_PAD = 0.15       # bbox 自身宽高的外扩比例
MAX_REGION_PAD = 0.05   # 单侧最多外扩全图 5%
MIN_REGION_PAD_PX = 24  # 单侧至少保留 24px 上下文
PASTE_MARGIN_PX = 12
VERIFY_CONFIDENCE = 0.65
MAX_REJECTIONS_PER_ISSUE = 2
FULL_IMAGE_AREA_THRESHOLD = 0.90
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLAN_DIR, exist_ok=True)

# Gemini 客户端：优先用 vertex 服务账号，否则走 GEMINI_API_KEY
if os.path.exists(KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
    client = genai.Client(vertexai=True, project="llm-6669", location="global")
    print("[INFO] Gemini 使用 vertex 服务账号")
else:
    client = genai.Client()
    print("[INFO] Gemini 使用 GEMINI_API_KEY")


def image_to_bytes(image_source):
    """支持从路径或 PIL Image 获取 PNG bytes，方便每轮把最新图发给 Gemini。"""
    if isinstance(image_source, (str, os.PathLike)):
        with open(image_source, "rb") as f:
            return f.read()
    buf = io.BytesIO()
    image_source.save(buf, format="PNG")
    return buf.getvalue()


def normalize_bbox(raw_bbox):
    """统一成 0~1 的 xyxy；兼容 Gemini 偶尔返回 0~1000 或 xywh。"""
    if not raw_bbox or len(raw_bbox) != 4:
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


def union_bbox(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def bbox_area(bb):
    return max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])


def clean_field(value, default=""):
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def build_instruction(e):
    instruction = clean_field(e.get("instruction"))
    if instruction:
        return instruction
    target = clean_field(e.get("target_object"), "target element")
    desired = clean_field(e.get("desired_state"))
    operation = clean_field(e.get("operation"), "Change")
    if desired:
        return f"{operation} only the {target} so that {desired}."
    return ""



def normalize_edit(e):
    if not isinstance(e, dict):
        return None

    bbox = normalize_bbox(e.get("bbox", []))
    source_bbox = normalize_bbox(e.get("source_bbox", []))
    target_bbox = normalize_bbox(e.get("target_bbox", []))
    edit_bbox = union_bbox([bbox, source_bbox, target_bbox])
    instruction = build_instruction(e)
    if edit_bbox is None or not instruction:
        return None

    target_object = clean_field(e.get("target_object"), "target element")
    return {
        "layout_type": clean_field(e.get("layout_type"), "other"),
        "operation": clean_field(e.get("operation"), "change"),
        "target_object": target_object,
        "failed_check": clean_field(e.get("failed_check"), target_object),
        "visual_signature": clean_field(e.get("visual_signature"), target_object),
        "reference_context": clean_field(e.get("reference_context")),
        "preserve": clean_field(
            e.get("preserve"),
            "all text, colors, styling, object identities, and unrelated elements",
        ),
        "current_state": clean_field(e.get("current_state"), "current incorrect layout"),
        "desired_state": clean_field(e.get("desired_state"), "required corrected layout"),
        "bbox": bbox,
        "source_bbox": source_bbox,
        "target_bbox": target_bbox,
        "edit_bbox": edit_bbox,
        "instruction": instruction,
    }



def compose_editor_prompt(e):
    """Deterministic fallback when crop-level Gemini prompting is unavailable."""
    operation = e.get("operation", "change").strip().capitalize()
    return (
        f"{operation} only the visible target so that {e['desired_state']}. "
        f"Preserve {e['preserve']}. Keep every other visible element unchanged."
    )


def gemini_crop_instruction(region_img, edit, retries=3):
    """Let Gemini inspect only the crop and write the instruction Qwen will see."""
    crop_bytes = image_to_bytes(region_img)
    prompt = f"""Inspect only the attached cropped image.

The required operation is: {edit['operation']}.
Required final layout: {edit['desired_state']}

Write one complete natural-language image-editing instruction for this crop.
Rules:
- Base the instruction only on visual content actually visible in this crop.
- Rewrite all target descriptions into crop-local visual language; do not copy full-image locators from the planning context.
- Mention only the source, destination, or guide that is visibly necessary inside the crop.
- Describe each necessary target, source, destination, or guide with enough visible crop-local cues to distinguish it from similar elements.
- State exactly one layout action.
- Explicitly preserve: {edit['preserve']}
- Explicitly name the actually visible nearby text, lines, blocks, borders, and symbols that must remain unchanged.
- For removal, require complete erasure and reconstruction of the immediately surrounding background, with no ghost, residue, outline, duplicate, replacement character, tiny text, or new mark.
- Do not mention bbox, coordinates, row/column numbers, ordering in the full image, or anything outside this crop.
- Do not include analysis or grounding explanations.
- Use a compact but sufficiently detailed paragraph; do not shorten the instruction by dropping target description or preservation details.
- End with: Keep every other visible element unchanged.
- Return STRICT JSON only: {{"instruction": "..."}}"""

    for attempt in range(retries):
        try:
            resp = metered_generate_content(
                client, os.environ['GEMINI_TOKEN_LOG'], model=GEMINI_MODEL,
                contents=[
                    types.Part(text=prompt),
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=crop_bytes)),
                ],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            raw = resp.text.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("no JSON object in crop-instruction response")
            instruction = clean_field(json.loads(match.group()).get("instruction"))
            if not instruction:
                raise ValueError("empty crop instruction")
            return instruction
        except Exception as ex:
            print(f"  [CROP PROMPT retry {attempt+1}/{retries}] {str(ex)[:120]}")
    raise RuntimeError("Crop-grounded prompt generation failed; refusing full-image fallback")


def gemini_plan(image_source, prompt, failed_questions, blocked_checks=None, rejection_history=None, retries=3):
    """Gemini 看当前图 + layout 失败问题 → 返回按优先级排序的结构化编辑计划。"""
    blocked_checks = blocked_checks or set()
    rejection_history = rejection_history or []
    blocked_text = "\n".join(f"- {q}" for q in sorted(blocked_checks)) or "None."
    rejection_text = "\n".join(f"- {x}" for x in rejection_history[-6:]) or "None."

    planner_prompt = f"""This image was generated from the prompt:
\"\"\"{prompt}\"\"\"

These self-discovered issues are BLOCKED because two candidate edits were rejected:
{blocked_text}

Recent rejected attempts and verifier reasons:
{rejection_text}

Independently inspect the CURRENT image against the original generation prompt.
Discover visible layout discrepancies yourself; do not assume an evaluator has identified them for you.
Plan minimal edits only for discrepancies visibly supported by the current image and original prompt.
Possible discrepancies may concern position, alignment, order, spacing, hierarchy, connection, direction, grouping, size, or relative placement.

Return STRICT JSON only: no markdown, no explanation, no extra text.
Use EXACTLY this list format:
[
  {{
    "layout_type": "position | alignment | order | spacing | hierarchy | connection | direction | grouping | size | other",
    "operation": "move | align | reorder | resize | connect | redirect | group | swap | other",
    "target_object": "the exact visual element or group to modify",
    "failed_check": "a concise description of the self-discovered visible discrepancy",
    "visual_signature": "current visible appearance and local-neighbor cues that uniquely locate the target",
    "reference_context": "visible alignment edge, destination, group, or guide needed for this action; empty if unnecessary",
    "preserve": "content and styling that must remain unchanged",
    "current_state": "the incorrect layout currently visible",
    "desired_state": "the exact corrected layout supported by the original prompt",
    "bbox": [x1, y1, x2, y2],
    "source_bbox": [x1, y1, x2, y2] or null,
    "target_bbox": [x1, y1, x2, y2] or null,
    "instruction": "one precise, self-contained layout editing command"
  }}
]

VISUAL GROUNDING RULES:
- failed_check MUST describe one discrepancy independently discovered from the current image and original prompt. Never select an item shown in BLOCKED checks.
- visual_signature MUST identify the target by how it looks NOW inside the crop. Use 2-4 observable cues such as object type, current color/shape, unique attached text or number, and a nearby local relationship.
- Do NOT rely only on a semantic category name, legend label, series name, or an ordinal such as "third from the left". Those labels or orders may already be wrong.
- instruction MUST be one atomic action on the visually identified target. Use "it" or "its" instead of repeating an unreliable semantic name.
- Do not combine independent actions. For example, changing a bar color and changing its height are two rounds. Return the single safest action first; the next round will inspect the updated image again.
- reference_context names the visible guide needed to perform the action. For a measured position or size, name the exact axis tick, gridline, edge, or neighboring object.
- bbox MUST include the complete target, all cues named in visual_signature, and the visible reference_context. Everything needed by the editor must be visible in this one crop.
- preserve MUST explicitly name the target properties and nearby content that must stay unchanged, excluding only the property changed by this action.

Precision rules:
- Sort the edits by priority: put the single most important and safest edit FIRST.
- The pipeline will apply ONLY the first edit, then ask you to re-plan from the updated image.
- If no visible layout discrepancy remains in the CURRENT image, return an empty list: [].
- All bbox values MUST use normalized [0,1] xyxy corner coordinates, not xywh and not 0~1000.
- bbox MUST cover the complete affected editing region.
- For movement, swapping, connection, or redirection, bbox MUST cover both the source area and destination area.
- source_bbox: the current location of the target element; use null if not applicable.
- target_bbox: the required destination location; use null if not applicable.
- target_object must identify the exact element or group, not just say "object" or "area".
- current_state and desired_state must be concrete and visually checkable.
- instruction must include target_object and desired_state, and must start with "Move only", "Align only", "Reorder only", "Resize only", "Connect only", "Swap only", "Change only", or "Remove".
- For connection or arrows: state the exact source, target, direction, and line/arrow style to preserve.
- For alignment or spacing: state the exact elements that must align and the reference edge or axis.
- For order or hierarchy: state the exact final order or layer relationship.
- One edit per layout issue; merge fixes within the same affected region.
- Layout corrections only; preserve all existing text, content, colors, icons, object identities, and visual style.
- Do not redesign the whole image."""

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

            # 空列表表示当前图已无剩余布局问题，是正常停止条件。
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


def save_plan_json(fname, it, img_path, save_path, failed, plan_info, round_idx, selected_edit, planned_count):
    plan_path = os.path.join(
        PLAN_DIR,
        fname.replace(".png", f"_round{round_idx}.json"),
    )
    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "round": round_idx,
        "file_name": fname,
        "domain": it.get("domain"),
        "dimension": it.get("dimension"),
        "id": it.get("id"),
        "source_image": img_path,
        "output_image": save_path,
        "original_prompt": it.get("prompt", ""),
        "failed_questions": failed,
        "planned_edit_count": planned_count,
        "selected_edit": selected_edit,
        **plan_info,
    }
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return plan_path


def crop_region(img, bbox, pad=REGION_PAD):
    W, H = img.size
    if bbox_area(bbox) >= FULL_IMAGE_AREA_THRESHOLD:
        box = (0, 0, W, H)
        return img.copy(), box

    # 按 bbox 尺寸动态外扩，避免旧版固定扩全图 15% 造成大范围误改。
    bbox_w = max(0.0, bbox[2] - bbox[0])
    bbox_h = max(0.0, bbox[3] - bbox[1])
    pad_x = min(MAX_REGION_PAD, max(MIN_REGION_PAD_PX / max(W, 1), bbox_w * pad))
    pad_y = min(MAX_REGION_PAD, max(MIN_REGION_PAD_PX / max(H, 1), bbox_h * pad))
    x1 = max(0.0, bbox[0] - pad_x); y1 = max(0.0, bbox[1] - pad_y)
    x2 = min(1.0, bbox[2] + pad_x); y2 = min(1.0, bbox[3] + pad_y)
    box = (int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))
    return img.crop(box), box


def paste_hard_bbox(bg, fg, crop_box, bbox, margin=PASTE_MARGIN_PX):
    """Hard-paste target bbox plus margin, clamped to the inspected crop."""
    cx1, cy1, cx2, cy2 = crop_box
    fg = fg.resize((cx2 - cx1, cy2 - cy1), Image.Resampling.LANCZOS)
    W, H = bg.size
    raw = (int(bbox[0] * W), int(bbox[1] * H),
           int(bbox[2] * W), int(bbox[3] * H))
    box = (max(cx1, raw[0] - margin), max(cy1, raw[1] - margin),
           min(cx2, raw[2] + margin), min(cy2, raw[3] + margin))
    local = (box[0] - cx1, box[1] - cy1, box[2] - cx1, box[3] - cy1)
    bg.paste(fg.crop(local), (box[0], box[1]))
    return bg


def edit_region(editor, region_img, editor_prompt, seed):
    return editor.edit(region_img, editor_prompt, seed)


def _pil_png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def verify_candidate(before_img, after_img, original_prompt, edit, failed_questions, retries=2):
    """比较编辑前后；只检查目标是否改善以及是否误伤无关内容。"""
    target_instruction = edit.get("editor_prompt", edit.get("instruction", ""))
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


def round_artifact_dir(fname, round_idx):
    stem = os.path.splitext(os.path.basename(fname))[0]
    path = os.path.join(PLAN_DIR, "intermediates", stem, f"round_{round_idx:02d}", "single")
    os.makedirs(path, exist_ok=True)
    return path


def save_round_image(image, directory, name):
    path = os.path.abspath(os.path.join(directory, name))
    image.convert("RGB").save(path)
    return path


def main():
    # 1. 读数据集，只留 layout 维度
    items = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                it = json.loads(line)
                if it.get("dimension") == TARGET_DIMENSION:
                    items.append(it)
    print(f"[INFO] {TARGET_DIMENSION} 维度共 {len(items)} 条")

    # 2. 筛出评测过且有 false 的图
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
    print("[LOAD] Loading configured editor backend ...")
    pipe = create_editor()
    print(f"[LOAD] {pipe.name} ready; steps={pipe.steps}.")

    # 4. 逐图处理：每轮只改一处，然后基于新图重新规划
    n_done, n_skip, n_fail = 0, 0, 0
    for i, (it, fname, img_path, failed) in enumerate(todo):
        save_path = os.path.join(OUTPUT_DIR, fname)
        if SKIP_EXISTING and os.path.exists(save_path):
            print(f"[{i+1}/{len(todo)}] SKIP {fname} (已存在)")
            n_skip += 1
            continue

        print(f"\n[{i+1}/{len(todo)}] {fname}  失败题数={len(failed)}")
        img = Image.open(img_path).convert("RGB")
        applied_rounds = 0
        rejection_counts = {}
        blocked_checks = set()
        rejection_history = []

        try:
            for round_idx in range(1, MAX_LAYOUT_ROUNDS + 1):
                print(f"  [ROUND {round_idx}/{MAX_LAYOUT_ROUNDS}] Gemini 基于当前图重新规划")
                edits, plan_info = gemini_plan(
                    img,
                    it.get("prompt", ""),
                    failed,
                    blocked_checks=blocked_checks,
                    rejection_history=rejection_history,
                )
                plan_info["blocked_checks"] = sorted(blocked_checks)
                plan_info["rejection_history"] = list(rejection_history)
                selected_edit = edits[0] if edits else None
                plan_path = save_plan_json(
                    fname, it, img_path, save_path, failed,
                    plan_info, round_idx, selected_edit, len(edits) if edits else 0,
                )
                print(f"  [PLAN JSON] {plan_path}")

                if edits is None:
                    print("  [STOP] Gemini 请求失败，停止当前图片")
                    break
                if len(edits) == 0:
                    print("  [STOP] Gemini 判断当前图没有剩余 layout 问题")
                    break

                e = edits[0]
                issue_key = (
                    e.get("failed_check")
                    or e.get("visual_signature")
                    or e.get("target_object")
                    or f"round-{round_idx}"
                )
                if issue_key in blocked_checks:
                    print(f"  [SKIP ISSUE] 已连续拒绝两次：{issue_key[:120]}")
                    continue
                print(
                    f"  [SELECT] {e['layout_type']}/{e['operation']} | "
                    f"{e['target_object']} | {e['current_state']} -> {e['desired_state']}"
                )
                region, box = crop_region(img, e["edit_bbox"])
                if region.size[0] < 32 or region.size[1] < 32:
                    print("  [STOP] 选中区域太小，停止当前图片")
                    break

                try:
                    e["editor_prompt"] = gemini_crop_instruction(region, e)
                except RuntimeError as ex:
                    print(f"  [STOP] {ex}")
                    break
                print(f"  [EDITOR PROMPT] {e['editor_prompt']}")
                before = img.copy()
                artifact_dir = round_artifact_dir(fname, round_idx)
                artifacts = {
                    "before_full": save_round_image(before, artifact_dir, "before_full.png"),
                    "input_crop": save_round_image(region, artifact_dir, "input_crop.png"),
                }
                e["artifact_dir"] = os.path.abspath(artifact_dir)
                save_plan_json(
                    fname, it, img_path, save_path, failed,
                    plan_info, round_idx, e, len(edits),
                )
                print(f"  [EDIT] edit_bbox={e['edit_bbox']} crop_box={box}")
                fixed = edit_region(
                    pipe,
                    region,
                    e["editor_prompt"],
                    SEED + i * 100 + round_idx,
                )
                artifacts["model_output_crop"] = save_round_image(fixed, artifact_dir, "model_output_crop.png")
                candidate = paste_hard_bbox(
                    before.copy(), fixed.convert("RGB"), box, e["edit_bbox"]
                )
                artifacts["candidate_full"] = save_round_image(candidate, artifact_dir, "candidate_full.png")
                accepted, verify_info = verify_candidate(
                    before, candidate, it.get("prompt", ""), e, failed
                )
                verify_info["editor_prompt"] = e["editor_prompt"]
                verify_info["accepted"] = bool(accepted)
                verify_info["artifact_dir"] = os.path.abspath(artifact_dir)
                artifacts["committed_full" if accepted else "rollback_full"] = save_round_image(
                    candidate if accepted else before, artifact_dir,
                    "committed_full.png" if accepted else "rollback_full.png",
                )
                verify_info["artifacts"] = artifacts
                verify_path = save_verification_json(fname, round_idx, verify_info)
                print(f"  [VERIFY] {verify_info['status']} | {verify_path}")
                if not accepted:
                    # img 始终是“最近一次通过验证的工作图”。这里只丢弃
                    # candidate，绝不回到最初原图，也不覆盖此前已接受修改。
                    reason = str(verify_info.get("reason", "no verifier reason"))
                    rejection_counts[issue_key] = rejection_counts.get(issue_key, 0) + 1
                    reject_count = rejection_counts[issue_key]
                    rejection_history.append(
                        f"{issue_key} | attempt={reject_count} | reason={reason}"
                    )
                    print(
                        f"  [ROLLBACK-LOCAL] 当前候选被拒绝，保留此前通过的工作图 "
                        f"({reject_count}/{MAX_REJECTIONS_PER_ISSUE})"
                    )
                    if reject_count >= MAX_REJECTIONS_PER_ISSUE:
                        blocked_checks.add(issue_key)
                        print(f"  [SKIP ISSUE] 本问题两次失败，下一轮处理其他问题：{issue_key[:120]}")
                    else:
                        print(f"  [RETRY ISSUE] 下一轮允许按验证原因换一种方案重试：{reason[:120]}")
                    continue
                img = candidate
                applied_rounds += 1

            if applied_rounds > 0:
                img.save(save_path)
                print(f"  [SAVE] {save_path}（完成 {applied_rounds} 轮单点编辑）")
                n_done += 1
            else:
                print("  [FAIL] 没有实际应用任何布局编辑，未保存结果")
                n_fail += 1

        except Exception as ex:
            print(f"  [ERROR] {str(ex)[:200]}")
            if applied_rounds > 0:
                img.save(save_path)
                print(f"  [SAVE-PARTIAL] {save_path}（保存前 {applied_rounds} 轮结果）")
                n_done += 1
            else:
                n_fail += 1


    print(f"\n[DONE] 成功 {n_done} / 跳过 {n_skip} / 失败 {n_fail}")
    print(f"[INFO] 输出目录：{OUTPUT_DIR}")
    print(f"[INFO] Gemini 计划 JSON：{PLAN_DIR}")


if __name__ == "__main__":
    main()
