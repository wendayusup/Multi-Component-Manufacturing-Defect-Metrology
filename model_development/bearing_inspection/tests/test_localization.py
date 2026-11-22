"""
Unit Tests for Defect Localization and Visualization Layer.
Tests Grad-CAM computation, bounding box extraction, and annotated image rendering.
"""
import unittest
from PIL import Image
import numpy as np

from app.model_loader import get_model_manager
from cnn.localization import locate_defects, generate_annotated_image_base64


class TestDefectLocalization(unittest.TestCase):

    def setUp(self):
        self.model_mgr = get_model_manager()

    def test_ok_image_has_no_defect_boxes(self):
        """Clean / normal casting image should not produce defect boxes."""
        import os
        ok_path = "dataset/casting_data/casting_data/test/ok_front/cast_ok_0_526.jpeg"
        if os.path.exists(ok_path):
            img = Image.open(ok_path)
        else:
            img = Image.new("RGB", (200, 200), color=(150, 150, 150))
            
        loc = locate_defects(self.model_mgr.model, img)
        if os.path.exists(ok_path):
            self.assertFalse(loc["has_defect"])
            self.assertEqual(len(loc["boxes"]), 0)
            self.assertEqual(len(loc["points"]), 0)
            annotated_b64 = generate_annotated_image_base64(img, loc)
            self.assertIsNone(annotated_b64)

    def test_localization_data_contract(self):
        """Tests schema structure of localization dictionary."""
        img = Image.new("RGB", (200, 200), color=(100, 100, 100))
        loc = locate_defects(self.model_mgr.model, img)
        self.assertIn("has_defect", loc)
        self.assertIn("defect_count", loc)
        self.assertIn("boxes", loc)
        self.assertIn("points", loc)
        self.assertIn("confidence", loc)


if __name__ == "__main__":
    unittest.main()
