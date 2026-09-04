"""Single-clean-crop editor backends used by all Agent1 dimensions."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

from PIL import Image


SENSE_DEFAULT_MODEL = "/mmu-vcg/zb08/zixuan/models/SenseNova-U1.5-8B-MoT"
SENSE_DEFAULT_REPO = "/mmu-vcg/zb08/zixuan/BIZ/SenseNova-U1"
QWEN_DEFAULT_MODEL = "/mmu-vcg/zb08/CKPTS/qwen-edit_2511"
SENSE_TARGET_PIXELS = 3456 * 1152


def _required_path(env_name: str, default: str) -> Path:
    path = Path(os.environ.get(env_name, default)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{env_name} path does not exist: {path}")
    return path


def _load_sensenova_module(repo: Path):
    entry = repo / "examples/editing/inference.py"
    if not entry.is_file():
        raise FileNotFoundError(entry)
    for path in (repo, repo / "src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location("bizgenagent_sensenova", entry)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load SenseNova entry: {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("SenseNovaU1Editing", "_resolve_output_size"):
        if not hasattr(module, name):
            raise RuntimeError(f"SenseNova entry lacks {name}: {entry}")
    return module


class SenseNovaEditor:
    name = "sensenova"
    steps = 50

    def __init__(self):
        import torch

        self.torch = torch
        self.repo = _required_path("BIZ_SENSENOVA_REPO", SENSE_DEFAULT_REPO)
        self.model = _required_path("BIZ_SENSENOVA_MODEL", SENSE_DEFAULT_MODEL)
        module = _load_sensenova_module(self.repo)
        self.resolve_output_size = module._resolve_output_size
        self.engine = module.SenseNovaU1Editing(
            str(self.model), device="cuda", dtype=torch.bfloat16, device_map="auto"
        )

    def edit(self, clean_crop: Image.Image, prompt: str, seed: int) -> Image.Image:
        source_size = clean_crop.size
        model_size = tuple(
            self.resolve_output_size(
                [clean_crop], explicit=None, target_pixels=SENSE_TARGET_PIXELS
            )
        )
        model_input = clean_crop.convert("RGB").resize(
            model_size, Image.Resampling.LANCZOS
        )
        print(
            f"  [EDITOR] SenseNova clean_crop={source_size} model_input={model_size} "
            f"steps={self.steps} seed={seed}",
            flush=True,
        )
        with self.torch.inference_mode():
            images, _thinking = self.engine.edit(
                prompt,
                [model_input],
                image_size=model_size,
                cfg_scale=4.0,
                img_cfg_scale=1.0,
                cfg_norm="none",
                timestep_shift=3.0,
                num_steps=self.steps,
                batch_size=1,
                think_mode=False,
                seed=seed,
            )
        if not images or len(images) != 1 or images[0] is None:
            raise RuntimeError("SenseNova must return exactly one image")
        return images[0].convert("RGB").resize(source_size, Image.Resampling.LANCZOS)


class QwenEditor:
    name = "qwen"
    steps = 40

    def __init__(self):
        import torch
        from diffusers import QwenImageEditPlusPipeline

        self.torch = torch
        self.model = _required_path("BIZ_QWEN_MODEL", QWEN_DEFAULT_MODEL)
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            str(self.model), torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to("cuda")

    def edit(self, clean_crop: Image.Image, prompt: str, seed: int) -> Image.Image:
        generator = self.torch.Generator(device="cuda").manual_seed(seed)
        print(
            f"  [EDITOR] Qwen clean_crop={clean_crop.size} steps={self.steps} seed={seed}",
            flush=True,
        )
        with self.torch.inference_mode():
            output = self.pipe(
                image=[clean_crop],
                prompt=prompt,
                generator=generator,
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=self.steps,
                guidance_scale=1.0,
                num_images_per_prompt=1,
            )
        image = output.images[0] if hasattr(output, "images") else output[0]
        return image.convert("RGB")


def create_editor():
    backend = os.environ.get("BIZ_EDITOR_BACKEND", "sensenova").strip().lower()
    if backend == "sensenova":
        return SenseNovaEditor()
    if backend == "qwen":
        return QwenEditor()
    raise ValueError("BIZ_EDITOR_BACKEND must be 'sensenova' or 'qwen'")
