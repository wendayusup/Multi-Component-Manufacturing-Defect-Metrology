"""
Unit Tests for Image Preprocessing Layer.
Verifies grayscale conversion, resizing, tensor shapes, and normalization scaling.
"""
import unittest
import numpy as np
from PIL import Image

from app.preprocessing import preprocess_pil_image
import config


class TestPreprocessing(unittest.TestCase):

    def test_rgb_to_normalized_tensor(self):
        """Tests that RGB PIL Image transforms into normalized (1, 1, H, W) tensor."""
        rgb_img = Image.new("RGB", (200, 150), color=(100, 150, 200))
        tensor = preprocess_pil_image(rgb_img, target_size=(64, 64), channels=1)

        self.assertEqual(tensor.shape, (1, 1, 64, 64))
        self.assertEqual(tensor.dtype, np.float32)
        # Pixel values must be scaled between 0.0 and 1.0
        self.assertGreaterEqual(tensor.min(), 0.0)
        self.assertLessEqual(tensor.max(), 1.0)

    def test_grayscale_input_support(self):
        """Tests that already-grayscale image transforms consistently."""
        gray_img = Image.new("L", (128, 128), color=128)
        tensor = preprocess_pil_image(gray_img, target_size=(64, 64), channels=1)

        self.assertEqual(tensor.shape, (1, 1, 64, 64))
        # Expected value is approximately 128 / 255.0
        expected = 128.0 / config.NORMALIZATION_SCALE
        self.assertAlmostEqual(float(tensor[0, 0, 32, 32]), expected, places=2)


if __name__ == "__main__":
    unittest.main()

