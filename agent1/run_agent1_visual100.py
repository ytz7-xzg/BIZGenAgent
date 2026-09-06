#!/usr/bin/env python3
"""Patch Agent1, run the 100-sample subset, assemble final images and score.

The existing agent source is upgraded in place under zixuan/BIZ.  Runtime
copies are written inside a new result directory, so data/output constants can
be redirected without touching BizGenEval or the historical result folders.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import html
import json
import os
import re
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


BIZ_ROOT = Path("/mmu-vcg/zb08/zixuan/BIZ")
# Run the exact code checked out from this repository.  Data and result paths
# remain under BIZ_ROOT, but no second mutable source copy is required.
AGENT1_DIR = Path(__file__).resolve().parent
TOOLS_DIR = BIZ_ROOT / "tools"
SAMPLE_ROOT = BIZ_ROOT / "results/sample100"
DATA_PATH = SAMPLE_ROOT / "sample100.jsonl"
REPO_ROOT = Path("/mmu-vcg/zb08/wps4.28/7-25-BizGen/BizGenEval")
DIMENSIONS = ("text", "knowledge", "attribute", "layout")


def say(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def replace_assignment(source: str, name: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^(?P<indent>\s*){re.escape(name)}\s*=.*$")
    replacement = lambda match: f'{match.group("indent")}{name} = {value!r}'
    updated, count = pattern.subn(replacement, source, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one assignment for {name}, found {count}")
    return updated


def item_name(item: dict) -> str:
    name = item.get("image") or item.get("image_name") or item.get("filename")
    if name:
        return Path(str(name)).name
    return f"{item['domain']}_{item['dimension']}_{item['id']}.png"


def load_expected_names(data_path: Path = DATA_PATH) -> list[str]:
    names: list[str] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            names.append(item_name(item))
    if not names or len(names) != len(set(names)):
        raise RuntimeError(f"Expected unique samples, found {len(names)} rows / {len(set(names))} names")
    return names


def find_original_dir(names: list[str]) -> Path:
    preferred = [
        SAMPLE_ROOT / "originals",
        SAMPLE_ROOT / "images_original",
        SAMPLE_ROOT / "original",
        SAMPLE_ROOT / "images",
    ]
    candidates = preferred + [p for p in SAMPLE_ROOT.iterdir() if p.is_dir()]
    seen: set[Path] = set()
    for directory in candidates:
        if directory in seen:
            continue
        seen.add(directory)
        if all((directory / name).is_file() for name in names):
            return directory
    raise FileNotFoundError(
        f"No directory under {SAMPLE_ROOT} contains all requested originals"
    )


def load_excluded_names(paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Exclude manifest missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    excluded.add(item_name(json.loads(line)))
    return excluded


def prepare_named_subset(result_root: Path, requested_names: list[str]) -> Path:
    """Write an exact manifest for explicitly requested image filenames."""
    requested = []
    seen: set[str] = set()
    for raw_name in requested_names:
        name = Path(raw_name).name
        if name in seen:
            raise ValueError(f"Duplicate --case-name: {name}")
        seen.add(name)
        requested.append(name)

    available: dict[str, dict] = {}
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                available[item_name(item)] = item

    missing = [name for name in requested if name not in available]
    if missing:
        raise FileNotFoundError(
            "Requested cases are absent from the canonical manifest: "
            + ", ".join(missing)
        )
    selected = [available[name] for name in requested]
    manifest = result_root / "cases_named.jsonl"
    manifest.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    (result_root / "selected_cases.json").write_text(
        json.dumps(
            [
                {"dimension": item["dimension"], "image": item_name(item)}
                for item in selected
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    say(f"Selected {len(selected)} explicitly named cases: {manifest}")
    return manifest


def prepare_case_subset(
    result_root: Path,
    cases_per_dimension: int,
    seed: int,
    exclude_manifests: list[str] | None = None,
) -> Path:
    """Select reproducible, actually-failing cases so every worker gets real work."""
    if cases_per_dimension <= 0:
        return DATA_PATH
    excluded = load_excluded_names(exclude_manifests or [])
    grouped = {dimension: [] for dimension in DIMENSIONS}
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            dimension = item.get("dimension")
            if dimension not in grouped:
                continue
            name = item_name(item)
            if name in excluded:
                continue
            mark_path = SAMPLE_ROOT / "mark_original" / name.replace(".png", ".json")
            if not mark_path.is_file():
                continue
            evaluation = json.loads(mark_path.read_text(encoding="utf-8"))
            has_failure = any(
                value.get("result") is False and value.get("raw_description")
                for value in evaluation.get("meta_info", {}).values()
            )
            if has_failure:
                grouped[dimension].append(item)

    rng = random.Random(seed)
    selected = []
    for dimension in DIMENSIONS:
        candidates = sorted(grouped[dimension], key=item_name)
        rng.shuffle(candidates)
        if len(candidates) < cases_per_dimension:
            raise RuntimeError(
                f"{dimension} has only {len(candidates)} eligible cases; "
                f"requested {cases_per_dimension}"
            )
        selected.extend(candidates[:cases_per_dimension])

    manifest = result_root / f"cases_{cases_per_dimension}x4_seed{seed}.jsonl"
    manifest.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    (result_root / "selected_cases.json").write_text(
        json.dumps(
            [
                {"dimension": item["dimension"], "image": item_name(item)}
                for item in selected
            ],
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    say(f"Selected {len(selected)} cases: {manifest}")
    if excluded:
        say(f"Excluded {len(excluded)} previously tested images")
    return manifest


def prepare_runtime_scripts(result_root: Path, data_path: Path) -> dict[str, Path]:
    runtime_dir = result_root / "runtime"
    edited_dir = result_root / "edited"
    plan_root = result_root / "plans"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    backend_source = AGENT1_DIR / "editor_backend.py"
    if not backend_source.is_file():
        raise FileNotFoundError(backend_source)
    shutil.copy2(backend_source, runtime_dir / backend_source.name)

    scripts: dict[str, Path] = {}
    for dimension in DIMENSIONS:
        source_path = AGENT1_DIR / f"{dimension}.py"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source = source_path.read_text(encoding="utf-8")
        if "CLEAN_CROP_PROMPT_V3 = True" not in source:
            raise RuntimeError(f"Clean-crop prompt pipeline is absent from {source_path}")

        source = replace_assignment(source, "DATA_PATH", str(data_path))
        source = replace_assignment(source, "OUTPUT_DIR", str(edited_dir))
        source = replace_assignment(source, "PLAN_DIR", str(plan_root / dimension))

        runtime_path = runtime_dir / f"{dimension}.py"
        runtime_path.write_text(source, encoding="utf-8")
        subprocess.run(
            [sys.executable, "-m", "py_compile", str(runtime_path)], check=True
        )
        scripts[dimension] = runtime_path
        say(f"Prepared {dimension}: {runtime_path}")
    return scripts


def worker_env(gpu: str, token_log: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{TOOLS_DIR}:{current_pythonpath}" if current_pythonpath else str(TOOLS_DIR)
    )
    env["GEMINI_TOKEN_LOG"] = str(token_log)
    env.setdefault("BIZ_EDITOR_BACKEND", "sensenova")
    env.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "/mmu-vcg/zb08/llm-6669-1b56d4a3712d.json",
    )
    env.setdefault("GOOGLE_CLOUD_PROJECT", "llm-6669")
    env.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    env.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    return env


def run_dimension(
    dimension: str, script: Path, gpu: str, result_root: Path
) -> tuple[str, int, float]:
    log_path = result_root / "logs" / f"{dimension}.log"
    token_log = result_root / "tokens" / f"{dimension}.jsonl"
    started = time.monotonic()
    say(f"START {dimension} GPU={gpu}; log={log_path}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== start {time.strftime('%F %T')} GPU={gpu} =====\n")
        log.flush()
        process = subprocess.run(
            [sys.executable, "-u", str(script)],
            cwd=str(script.parent),
            env=worker_env(gpu, token_log),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.monotonic() - started
    say(f"EXIT {dimension} code={process.returncode} seconds={elapsed:.1f}")
    return dimension, process.returncode, elapsed


def run_generation(
    scripts: dict[str, Path], gpu_ids: list[str], result_root: Path
) -> dict[str, dict[str, float | int | str]]:
    gpu_map = dict(zip(DIMENSIONS, gpu_ids, strict=True))
    status: dict[str, dict[str, float | int | str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                run_dimension, dimension, scripts[dimension], gpu_map[dimension], result_root
            ): dimension
            for dimension in DIMENSIONS
        }
        for future in concurrent.futures.as_completed(futures):
            dimension, returncode, elapsed = future.result()
            status[dimension] = {
                "gpu": gpu_map[dimension],
                "returncode": returncode,
                "wall_seconds": round(elapsed, 3),
            }
    return status


def build_final(
    names: list[str], original_dir: Path, result_root: Path
) -> tuple[int, int]:
    edited_dir = result_root / "edited"
    final_dir = result_root / "final"
    before_dir = result_root / "before"
    final_dir.mkdir(parents=True, exist_ok=True)
    before_dir.mkdir(parents=True, exist_ok=True)
    for stale in final_dir.glob("*.png"):
        stale.unlink()
    for stale in before_dir.glob("*.png"):
        stale.unlink()

    edited = 0
    fallback = 0
    missing: list[str] = []
    for name in names:
        generated_path = edited_dir / name
        original_path = original_dir / name
        if generated_path.is_file():
            source = generated_path
            edited += 1
        elif original_path.is_file():
            source = original_path
            fallback += 1
        else:
            missing.append(name)
            continue
        shutil.copy2(original_path, before_dir / name)
        shutil.copy2(source, final_dir / name)

    if missing:
        raise FileNotFoundError(
            f"Cannot assemble final: {len(missing)} samples have neither edit nor original: "
            + ", ".join(missing[:10])
        )
    final_count = len(list(final_dir.glob("*.png")))
    if final_count != len(names):
        raise RuntimeError(
            f"Final directory contains {final_count} PNGs, expected {len(names)}"
        )
    return edited, fallback


def build_comparison_web(data_path: Path, result_root: Path) -> Path:
    rows = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            name = item_name(item)
            dimension = html.escape(str(item.get("dimension", "unknown")))
            safe_name = html.escape(name)
            rows.append(
                f'<section><h2>{dimension} · {safe_name}</h2>'
                f'<div class="pair"><figure><figcaption>Before</figcaption>'
                f'<a href="before/{safe_name}"><img src="before/{safe_name}"></a></figure>'
                f'<figure><figcaption>Agent Final</figcaption>'
                f'<a href="final/{safe_name}"><img src="final/{safe_name}"></a></figure>'
                f'</div><p><a href="logs/{dimension}.log">worker log</a></p></section>'
            )
    page = """<!doctype html><meta charset="utf-8"><title>Agent1 SenseNova smoke4</title>
<style>body{font:16px system-ui;margin:24px;background:#f4f6f8;color:#172033}section{background:white;padding:18px;margin:0 0 22px;border-radius:12px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}figcaption{font-weight:700;margin-bottom:8px}img{width:100%;max-height:72vh;object-fit:contain;background:#eef1f4;border:1px solid #ccd3dc}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<h1>Agent1 · SenseNova · 4-case smoke test</h1>""" + "".join(rows)
    path = result_root / "comparison.html"
    path.write_text(page, encoding="utf-8")
    say(f"Comparison webpage: {path}")
    return path


def run_logged(command: list[str], cwd: Path, log) -> None:
    say("RUN " + " ".join(command))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    # Keep scoring on the same Vertex credentials as generation.  Without
    # these defaults the evaluator can silently write all-zero marks after an
    # ADC failure, which happened in the earlier V2 run.
    env.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "/mmu-vcg/zb08/llm-6669-1b56d4a3712d.json",
    )
    env.setdefault("GOOGLE_CLOUD_PROJECT", "llm-6669")
    env.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    env.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command exited with {result.returncode}: {' '.join(command)}"
        )


def score(result_root: Path, data_path: Path, expected_count: int, force_score: bool) -> None:
    final_dir = result_root / "final"
    mark_dir = result_root / "mark"
    summary_dir = result_root / "summary"
    marker = result_root / "scoring.done"
    if marker.exists() and not force_score:
        say("Scoring marker exists; use --force-score to score again")
        return

    mark_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    credential_path = Path(
        os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS",
            "/mmu-vcg/zb08/llm-6669-1b56d4a3712d.json",
        )
    )
    if not credential_path.is_file():
        raise FileNotFoundError(
            "Gemini/Vertex credential file is missing; scoring was not started: "
            f"{credential_path}"
        )
    score_log = result_root / "logs/scoring.log"
    with score_log.open("a", encoding="utf-8") as log:
        evaluation_command = [
            sys.executable,
            "-m",
            "evaluation.image_evaluation",
            "--data_path",
            str(data_path),
            "--img_dir",
            str(final_dir),
            "--save_dir",
            str(mark_dir),
        ]
        if force_score:
            evaluation_command.append("--force_rerun")
        run_logged(evaluation_command, REPO_ROOT, log)

        mark_files = list(mark_dir.glob("*.json"))
        missing_answers = sum(
            path.read_text(encoding="utf-8", errors="replace").count(
                '"reason": "missing_from_output"'
            )
            for path in mark_files
        )
        if len(mark_files) != expected_count or missing_answers:
            raise RuntimeError(
                "Evaluation output is incomplete; summary was intentionally stopped. "
                f"mark_json={len(mark_files)}, missing_from_output={missing_answers}. "
                f"See {score_log}"
            )
        run_logged(
            [
                sys.executable,
                "-m",
                "evaluation.summarize",
                "--data_path",
                str(data_path),
                "--result_dir",
                str(mark_dir),
                "--save_dir",
                str(summary_dir),
            ],
            REPO_ROOT,
            log,
        )

    marker.write_text(
        json.dumps(
            {
                "finished_at": time.strftime("%F %T"),
                "mark_json": len(list(mark_dir.glob("*.json"))),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    say(f"Scoring completed: {summary_dir}")


def print_csv(result_root: Path) -> None:
    for name in ("summary_by_dimension.csv", "summary_by_domain.csv"):
        path = result_root / "summary" / name
        print(f"\n===== {name} =====", flush=True)
        if path.is_file():
            print(path.read_text(encoding="utf-8"), end="", flush=True)
        else:
            print("missing", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        default="0,2,3,4",
        help="Four unique GPU IDs for text,knowledge,attribute,layout",
    )
    parser.add_argument(
        "--result-name",
        default="agent1_visual100",
        help="Single directory name created under zixuan/BIZ/results",
    )
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--force-score", action="store_true")
    parser.add_argument(
        "--cases-per-dimension",
        type=int,
        default=0,
        help="Use a reproducible eligible subset per dimension; 0 runs the full manifest",
    )
    parser.add_argument("--case-seed", type=int, default=42)
    parser.add_argument(
        "--case-name",
        action="append",
        default=[],
        help="Run one exact image filename from the canonical manifest; repeatable",
    )
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="JSONL manifest whose image names must not be sampled; repeatable",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.result_name):
        raise SystemExit("--result-name must be one simple directory name")
    gpu_ids = [part.strip() for part in args.gpus.split(",") if part.strip()]
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise SystemExit("--gpus must contain four unique GPU IDs")
    if args.cases_per_dimension < 0:
        raise SystemExit("--cases-per-dimension must be non-negative")
    if args.case_name and args.cases_per_dimension:
        raise SystemExit("--case-name cannot be combined with --cases-per-dimension")

    required = [DATA_PATH, REPO_ROOT, AGENT1_DIR]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))

    result_root = BIZ_ROOT / "results" / args.result_name
    for path in (
        result_root / "edited",
        result_root / "final",
        result_root / "logs",
        result_root / "plans",
        result_root / "tokens",
        result_root / "mark",
        result_root / "summary",
    ):
        path.mkdir(parents=True, exist_ok=True)
    for dimension in DIMENSIONS:
        (result_root / "plans" / dimension).mkdir(parents=True, exist_ok=True)

    lock_path = result_root / "controller.lock"
    lock = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Another controller is already using {result_root}")
    lock.write(str(os.getpid()) + "\n")
    lock.flush()

    all_names = load_expected_names(DATA_PATH)
    original_dir = find_original_dir(all_names)
    if args.case_name:
        active_data_path = prepare_named_subset(result_root, args.case_name)
    else:
        active_data_path = prepare_case_subset(
            result_root,
            args.cases_per_dimension,
            args.case_seed,
            args.exclude_manifest,
        )
    names = load_expected_names(active_data_path)
    say(f"Original directory: {original_dir}")
    say(f"Result directory: {result_root}")

    scripts = prepare_runtime_scripts(result_root, active_data_path)
    status = run_generation(scripts, gpu_ids, result_root)
    failed = {
        name: info
        for name, info in status.items()
        if int(info["returncode"]) != 0
    }
    generation_record = {
        "finished_at": time.strftime("%F %T"),
        "workers": status,
    }
    (result_root / "generation_workers.json").write_text(
        json.dumps(generation_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failed:
        raise SystemExit(
            "Generation worker failure; final assembly and scoring were stopped: "
            + json.dumps(failed, ensure_ascii=False)
        )

    edited, fallback = build_final(names, original_dir, result_root)
    build_comparison_web(active_data_path, result_root)
    generation_record.update(
        {"edited_images": edited, "fallback_originals": fallback, "final_images": len(names)}
    )
    (result_root / "generation.done").write_text(
        json.dumps(generation_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    say(f"Generation complete: edited={edited}, fallback={fallback}, final={len(names)}")

    if args.generation_only:
        return
    score(result_root, active_data_path, len(names), args.force_score)
    print_csv(result_root)


if __name__ == "__main__":
    main()
