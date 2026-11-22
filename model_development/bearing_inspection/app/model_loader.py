"""
Model Loading and Management Layer.
Loads trained pure-NumPy CNN weights and architecture.
Gracefully handles missing, corrupt, or mismatched weights without crashing.
Allows injecting test models for automated testing.
"""
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np

import config
from cnn.model import Sequential, build_casting_cnn


class CastingModelManager:
    """
    Manages lifecycle and loading of the Casting Defect CNN model.
    """
    def __init__(
        self,
        weights_file: Optional[Path] = None,
        meta_file: Optional[Path] = None,
        auto_load: bool = True
    ):
        self.weights_file = Path(weights_file or config.DEFAULT_WEIGHTS_FILE)
        self.meta_file = Path(meta_file or config.DEFAULT_META_FILE)
        self.model: Optional[Sequential] = None
        self.metadata: Dict = {}
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None

        if auto_load:
            self.load()

    def load(self) -> bool:
        """
        Attempts to build model architecture and load trained weights.
        Returns True if successful, False otherwise.
        """
        self.is_loaded = False
        self.load_error = None
        self.model = None

        if not self.weights_file.exists():
            self.load_error = (
                f"File bobot model '{self.weights_file.name}' tidak ditemukan di folder weights/. "
                "Silakan jalankan offline training (cnn/train.py) terlebih dahulu."
            )
            return False

        try:
            # Build baseline architecture
            model = build_casting_cnn(
                input_shape=(config.IMAGE_CHANNELS, config.IMAGE_TARGET_SIZE[1], config.IMAGE_TARGET_SIZE[0]),
                num_classes=len(config.CLASS_NAMES)
            )
            # Load weights
            model.load_weights(self.weights_file)
            self.model = model

            # Load metadata if exists
            if self.meta_file.exists():
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {
                    "model_version": config.MODEL_VERSION,
                    "module": "casting_defect",
                    "class_names": config.CLASS_NAMES
                }

            self.is_loaded = True
            return True
        except Exception as e:
            self.load_error = f"Gagal memuat model weights: {str(e)}"
            self.model = None
            self.is_loaded = False
            return False

    def reload(self) -> bool:
        """Forces reloading the weights and metadata from disk."""
        return self.load()

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Runs inference on the preprocessed tensor.
        Returns softmax probability array of shape (N, num_classes).
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(self.load_error or "Model belum dimuat.")

        return self.model.predict_proba(input_tensor)

    def set_test_model(self, test_model: Sequential, version: str = "test-cnn-fixture"):
        """Injects a mock/test model instance for unit tests."""
        self.model = test_model
        self.is_loaded = True
        self.load_error = None
        self.metadata = {
            "model_version": version,
            "module": "casting_defect",
            "class_names": config.CLASS_NAMES
        }


# Global model manager instance
_global_model_manager: Optional[CastingModelManager] = None


def get_model_manager() -> CastingModelManager:
    """Returns singleton model manager instance."""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = CastingModelManager()
    return _global_model_manager

