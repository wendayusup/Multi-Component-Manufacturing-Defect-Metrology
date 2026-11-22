"""
Unit Tests for Image Quality Assessment Layer.
Tests dimension restrictions, brightness checks, and sharpness/blur metrics.
"""
import unittest
import numpy as np
from PIL import Image

from app.quality_check import check_image_quality, compute_blur_proxy
import config


class TestQualityCheck(unittest.TestCase):

    def test_dimensions_below_minimum_rejected(self):
        """TEST-007: Image below minimum dimensions fails quality gate."""
        tiny_img = Image.new("RGB", (50, 50), color=(128, 128, 128))
        result = check_image_quality(tiny_img)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["passed"])
        self.assertTrue(any("terlalu kecil" in w.lower() for w in result["warnings"]))

    def test_adequate_dimensions_and_lighting_passes(self):
        """Normal quality casting image passes quality check."""
        # Create a textured image to produce normal Laplacian variance
        arr = np.random.randint(60, 190, size=(300, 300, 3), dtype=np.uint8)
        normal_img = Image.fromarray(arr)
        result = check_image_quality(normal_img)

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["warnings"]), 0)

    def test_extreme_darkness_flagged(self):
        """TEST-008: Extremely dark/underexposed image triggers review recommendation."""
        dark_img = Image.new("RGB", (200, 200), color=(10, 10, 10))
        result = check_image_quality(dark_img)

        self.assertEqual(result["status"], "review_recommended")
        self.assertTrue(any("gelap" in w.lower() for w in result["warnings"]))

    def test_extreme_overexposure_flagged(self):
        """TEST-008: Blown-out/overexposed image triggers review recommendation."""
        bright_img = Image.new("RGB", (200, 200), color=(250, 250, 250))
        result = check_image_quality(bright_img)

        self.assertEqual(result["status"], "review_recommended")
        self.assertTrue(any("terang" in w.lower() or "silau" in w.lower() for w in result["warnings"]))

    def test_extreme_blur_proxy(self):
        """Uniform / completely blurred image produces low Laplacian variance."""
        flat_arr = np.full((120, 120), 128.0, dtype=np.float32)
        blur_score = compute_blur_proxy(flat_arr)
        self.assertEqual(blur_score, 0.0)


if __name__ == "__main__":
    unittest.main()

