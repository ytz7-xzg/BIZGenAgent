import ast
from pathlib import Path
import unittest


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
                ["pipe", "region_img", "qwen_prompt", "seed"],
            )
            self.assertIn('"image": [region_img]', source)
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


if __name__ == "__main__":
    unittest.main()
