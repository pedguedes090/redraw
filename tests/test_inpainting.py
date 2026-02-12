"""
Tests for the Redraw inpainting pipeline.
Uses synthetic test images — no external fixtures required.
"""
import os
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_rgb_image(h=64, w=64):
    """Create a simple RGB test image with a coloured square."""
    img = np.full((h, w, 3), [100, 150, 200], dtype=np.uint8)
    img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = [255, 0, 0]
    return img


def _make_mask(h=64, w=64):
    """White square in the centre (area to inpaint)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255
    return mask


# ---------------------------------------------------------------------------
# AOT-GAN tests
# ---------------------------------------------------------------------------
class TestAOTGAN:
    """Tests for the AOT-GAN natural-image inpainting backend."""

    @pytest.fixture(autouse=True)
    def _load_model(self):
        from app import AOTGANInpainter, find_aot_model

        model_path = find_aot_model()
        if model_path is None:
            pytest.skip("AOT-GAN model not found in checkpoints/aot_gan/")
        self.inpainter = AOTGANInpainter()
        self.inpainter.load(model_path)

    def test_inpaint_produces_output(self):
        img = _make_rgb_image()
        mask = _make_mask()
        result = self.inpainter.inpaint(img, mask)
        assert isinstance(result, np.ndarray)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_output_dimensions_match_input(self):
        for size in [(64, 64), (128, 100), (200, 300)]:
            img = _make_rgb_image(*size)
            mask = _make_mask(*size)
            result = self.inpainter.inpaint(img, mask)
            assert result.shape[:2] == size

    def test_output_differs_from_input(self):
        img = _make_rgb_image()
        mask = _make_mask()
        result = self.inpainter.inpaint(img, mask)
        assert not np.array_equal(result, img), "Output should differ from the input"


# ---------------------------------------------------------------------------
# Mask-extraction tests (no model needed)
# ---------------------------------------------------------------------------
class TestMaskExtraction:
    """Tests for the mask extraction logic used by the Gradio UI."""

    def test_extract_rgba_layer(self):
        from app import extract_mask_from_editor

        bg = _make_rgb_image()
        layer = np.zeros((64, 64, 4), dtype=np.uint8)
        layer[10:20, 10:20, 3] = 255  # alpha channel marks drawn area
        input_dict = {"background": bg, "layers": [layer], "composite": None}
        image, mask = extract_mask_from_editor(input_dict)
        assert image is not None
        assert mask is not None
        assert mask[15, 15] == 255
        assert mask[0, 0] == 0

    def test_extract_none_returns_none(self):
        from app import extract_mask_from_editor

        image, mask = extract_mask_from_editor(None)
        assert image is None
        assert mask is None

    def test_extract_empty_layers(self):
        from app import extract_mask_from_editor

        bg = _make_rgb_image()
        input_dict = {"background": bg, "layers": [], "composite": None}
        image, mask = extract_mask_from_editor(input_dict)
        assert image is not None
        assert mask is not None
        assert mask.max() == 0  # no drawing → empty mask


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
class TestCLI:
    """Smoke-test the CLI entry point."""

    def test_cli_aot_end_to_end(self):
        from app import find_aot_model

        model_path = find_aot_model()
        if model_path is None:
            pytest.skip("AOT-GAN model not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "input.png")
            mask_path = os.path.join(tmpdir, "mask.png")
            out_path = os.path.join(tmpdir, "output.png")

            Image.fromarray(_make_rgb_image()).save(img_path)
            Image.fromarray(_make_mask()).save(mask_path)

            import subprocess

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cli.inpaint",
                    "--mode",
                    "aot",
                    "--image",
                    img_path,
                    "--mask",
                    mask_path,
                    "--output",
                    out_path,
                    "--model",
                    model_path,
                    "--seed",
                    "42",
                ],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.exists(out_path), "Output file not created"

            out_img = Image.open(out_path)
            assert out_img.size == (64, 64)


# ---------------------------------------------------------------------------
# Model-finder tests (no model needed)
# ---------------------------------------------------------------------------
class TestModelFinders:
    """Test that model-discovery helpers prefer the right files."""

    def test_find_aot_model_prefers_generator(self):
        from app import find_aot_model

        path = find_aot_model()
        if path is not None:
            assert os.path.basename(path).startswith("G"), (
                f"find_aot_model should return the Generator, got {path}"
            )
