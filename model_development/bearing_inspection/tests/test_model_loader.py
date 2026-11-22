"""
Unit Tests for Model Loader and Weight Management.
Verifies missing model weights handling, corrupted weights rejection,
and test model fixture injection.
"""
import unittest
from pathlib import Path
import numpy as np

from app.model_loader import CastingModelManager
from cnn.model import build_casting_cnn
import config


class TestModelLoader(unittest.TestCase):

    def test_missing_weights_handled_gracefully(self):
        """TEST-009: Missing weights file reports model_unavailable without crashing."""
        non_existent = Path("weights/non_existent_weights_xyz.npz")
        mgr = CastingModelManager(weights_file=non_existent, auto_load=True)

        self.assertFalse(mgr.is_loaded)
        self.assertIsNotNone(mgr.load_error)
        self.assertIn("tidak ditemukan", mgr.load_error.lower())

        with self.assertRaises(RuntimeError):
            dummy_tensor = np.zeros((1, 1, 64, 64), dtype=np.float32)
            mgr.predict(dummy_tensor)

    def test_incompatible_weights_file_rejected(self):
        """TEST-010: Weights archive with shape mismatch is rejected safely."""
        corrupt_path = Path("weights/corrupt_test_weights.npz")
        try:
            # Save corrupted weights with wrong dimensions
            np.savez(corrupt_path, layer_0_param_0=np.zeros((10, 10, 10)))
            mgr = CastingModelManager(weights_file=corrupt_path, auto_load=True)

            self.assertFalse(mgr.is_loaded)
            self.assertIsNotNone(mgr.load_error)
        finally:
            if corrupt_path.exists():
                corrupt_path.unlink()

    def test_test_model_injection(self):
        """TEST-011: Mock model fixture can be injected for automated testing."""
        mgr = CastingModelManager(auto_load=False)
        test_model = build_casting_cnn(input_shape=(1, 64, 64), num_classes=2)
        mgr.set_test_model(test_model, version="casting-cnn-test-fixture")

        self.assertTrue(mgr.is_loaded)
        self.assertEqual(mgr.metadata["model_version"], "casting-cnn-test-fixture")

        dummy_tensor = np.zeros((1, 1, 64, 64), dtype=np.float32)
        probs = mgr.predict(dummy_tensor)
        self.assertEqual(probs.shape, (1, 2))
        np.testing.assert_almost_equal(np.sum(probs), 1.0)


if __name__ == "__main__":
    unittest.main()

