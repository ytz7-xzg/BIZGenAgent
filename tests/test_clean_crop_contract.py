import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
import unittest

from PIL import Image

from agent1.editor_backend import SenseNovaEditor


ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ("text", "knowledge", "attribute", "layout")


class CleanCropContractTest(unittest.TestCase):
    def test_marker_pipeline_is_removed(self):
        self.assertFalse((ROOT / "agent1/repair_visual_utils.py").exists())
        forbidden = (
            "make_marked_crop",
            "marked_img",
            "marked_crop",
            "magenta rectangle",
            "image 2 marks the target",
        )
        for name in DIMENSIONS:
            source = (ROOT / "agent1" / f"{name}.py").read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token.lower(), source.lower(), f"{name}: {token}")

    def test_editor_receives_one_clean_crop(self):
        for name in DIMENSIONS:
            path = ROOT / "agent1" / f"{name}.py"
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            functions = {
                node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
            }
            edit_region = functions["edit_region"]
            self.assertEqual(
                [arg.arg for arg in edit_region.args.args],
                ["editor", "region_img", "editor_prompt", "seed"],
            )
            self.assertIn("return editor.edit(region_img, editor_prompt, seed)", source)
            self.assertIn("CLEAN_CROP_PROMPT_V3 = True", source)

    def test_crop_is_bbox_grounded_with_context(self):
        for name in DIMENSIONS:
            source = (ROOT / "agent1" / f"{name}.py").read_text(encoding="utf-8")
            self.assertIn("def crop_region(img, bbox, pad=REGION_PAD):", source)
            self.assertIn("MIN_REGION_PAD_PX", source)
            self.assertIn("MAX_REGION_PAD", source)

    def test_text_heavy_crops_have_reversible_orientation(self):
        for name in ("text", "knowledge"):
            source = (ROOT / "agent1" / f"{name}.py").read_text(encoding="utf-8")
            self.assertIn('"crop_rotation": crop_rotation', source)
            self.assertIn("def orient_crop_for_editor", source)
            self.assertIn("def restore_crop_orientation", source)
            self.assertIn("editor_region = orient_crop_for_editor", source)

    def test_sensenova_is_primary_editor(self):
        source = (ROOT / "agent1/editor_backend.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("BIZ_EDITOR_BACKEND", "sensenova")', source)
        self.assertIn("class SenseNovaEditor", source)
        self.assertIn("steps = 50", source)
        self.assertIn("SENSE_TARGET_PIXELS = 3456 * 1152", source)
        self.assertIn("[model_input]", source)

    def test_sensenova_adapter_uses_tested_size_and_settings(self):
        calls = {}

        class FakeEngine:
            def edit(self, prompt, images, **settings):
                calls.update(prompt=prompt, images=images, settings=settings)
                return [images[0].copy()], "mock"

        editor = SenseNovaEditor.__new__(SenseNovaEditor)
        editor.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        editor.resolve_output_size = lambda images, explicit, target_pixels: (3456, 1152)
        editor.engine = FakeEngine()
        source = Image.new("RGB", (300, 100), "white")
        result = editor.edit(source, "change one word", 42)

        self.assertEqual(result.size, source.size)
        self.assertEqual(calls["images"][0].size, (3456, 1152))
        self.assertEqual(calls["settings"]["image_size"], (3456, 1152))
        self.assertEqual(calls["settings"]["num_steps"], 50)
        self.assertEqual(calls["settings"]["seed"], 42)
        self.assertEqual(calls["settings"]["cfg_scale"], 4.0)
        self.assertEqual(calls["settings"]["img_cfg_scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
