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
import json
import os
import re
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


def load_expected_names() -> list[str]:
    names: list[str] = []
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            name = item.get("image") or item.get("image_name") or item.get("filename")
            if name:
                names.append(Path(str(name)).name)
                continue
            names.append(
                f"{item['domain']}_{item['dimension']}_{item['id']}.png"
            )
    if len(names) != 100 or len(set(names)) != 100:
        raise RuntimeError(
            f"Expected 100 unique samples, found {len(names)} rows / {len(set(names))} names"
        )
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
        f"No directory under {SAMPLE_ROOT} contains all 100 sampled originals"
    )


def prepare_runtime_scripts(result_root: Path) -> dict[str, Path]:
    runtime_dir = result_root / "runtime"
    edited_dir = result_root / "edited"
    plan_root = result_root / "plans"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    scripts: dict[str, Path] = {}
    for dimension in DIMENSIONS:
        source_path = AGENT1_DIR / f"{dimension}.py"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source = source_path.read_text(encoding="utf-8")
        if "CLEAN_CROP_PROMPT_V3 = True" not in source:
            raise RuntimeError(f"Clean-crop prompt pipeline is absent from {source_path}")

        source = replace_assignment(source, "DATA_PATH", str(DATA_PATH))
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
    final_dir.mkdir(parents=True, exist_ok=True)
    for stale in final_dir.glob("*.png"):
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
        shutil.copy2(source, final_dir / name)

    if missing:
        raise FileNotFoundError(
            f"Cannot assemble final: {len(missing)} samples have neither edit nor original: "
            + ", ".join(missing[:10])
        )
    final_count = len(list(final_dir.glob("*.png")))
    if final_count != 100:
        raise RuntimeError(f"Final directory contains {final_count} PNGs, expected 100")
    return edited, fallback


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


def score(result_root: Path, force_score: bool) -> None:
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
            str(DATA_PATH),
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
        if len(mark_files) != 100 or missing_answers:
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
                str(DATA_PATH),
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
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.result_name):
        raise SystemExit("--result-name must be one simple directory name")
    gpu_ids = [part.strip() for part in args.gpus.split(",") if part.strip()]
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise SystemExit("--gpus must contain four unique GPU IDs")

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

    names = load_expected_names()
    original_dir = find_original_dir(names)
    say(f"Original directory: {original_dir}")
    say(f"Result directory: {result_root}")

    scripts = prepare_runtime_scripts(result_root)
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
    generation_record.update(
        {"edited_images": edited, "fallback_originals": fallback, "final_images": 100}
    )
    (result_root / "generation.done").write_text(
        json.dumps(generation_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    say(f"Generation complete: edited={edited}, fallback={fallback}, final=100")

    if args.generation_only:
        return
    score(result_root, args.force_score)
    print_csv(result_root)


if __name__ == "__main__":
    main()
