"""
BEARING / CASTING DEFECT INSPECTION ADAPTER
-------------------------------------------
Integrates the Casting Defect Detector module (Casting_Detection folder):
- Pure-NumPy CNN architecture (64x64)
- Quality gating (blur score, brightness, dimensions)
- Sensitivity policy (Standard / High / Ultra)
- Weakly-supervised Grad-CAM bounding boxes & hotspot points
- Anomaly matrix rendering
"""
import sys
import io
import time
import importlib
from pathlib import Path
from typing import Dict, Any
from PIL import Image

# Ensure bearing_inspection root is accessible
UNIFIED_ROOT = Path(__file__).resolve().parent.parent.parent
CASTING_DIR = UNIFIED_ROOT / "model_development" / "bearing_inspection"
if not CASTING_DIR.exists():
    CASTING_DIR = UNIFIED_ROOT / "model_development" / "bearing_casting"
if not CASTING_DIR.exists():
    CASTING_DIR = UNIFIED_ROOT.parent / "Casting_Detection"
if str(CASTING_DIR) not in sys.path:
    sys.path.insert(0, str(CASTING_DIR))

from .base import BaseInspectionAdapter


class BearingAdapter(BaseInspectionAdapter):
    module_id = "bearing"
    display_name = "Casting / Bearing Inspection"
    category = "Casting Metal"

    def __init__(self):
        self.is_ready = False
        self.load_error = None
        self.metadata = {}
        self.load()

    def load(self) -> None:
        try:
            model_loader = importlib.import_module("app.model_loader")
            mgr = model_loader.get_model_manager()
            if mgr.is_loaded:
                self.is_ready = True
                self.metadata = mgr.metadata
            else:
                self.is_ready = False
                self.load_error = mgr.load_error or "Model belum dimuat."
        except Exception as e:
            self.is_ready = False
            self.load_error = str(e)

    def health(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "display_name": self.display_name,
            "category": self.category,
            "status": "ready" if self.is_ready else "error",
            "model_version": self.metadata.get("model_version", "casting-cnn-0.1.0"),
            "input_resolution": "64x64 Grayscale (Pure NumPy)",
            "accuracy": "92.80%",
            "error_message": self.load_error,
            "capabilities": ["classification", "quality_check", "grad_cam_boxes", "anomaly_matrix"]
        }

    def inspect(self, image_bytes: bytes, sensitivity: str = "standard", **kwargs) -> Dict[str, Any]:
        if not self.is_ready:
            return {
                "status": "model_unavailable",
                "error_message": self.load_error or "Model Casting belum siap."
            }

        try:
            inference_service = importlib.import_module("app.inference_service")
            result = inference_service.run_casting_inspection(
                image_bytes=image_bytes,
                original_filename=kwargs.get("filename", "input.jpg"),
                sensitivity=sensitivity
            )

            # Standardize output structure
            is_defect = (result.get("predicted_label") == "Defective")
            pred_label = result.get("predicted_label", "Unknown")
            conf = float(result.get("confidence_score") or 0.0) * 100.0

            loc = result.get("defect_localization", {})
            boxes = loc.get("boxes", [])

            import base64 as b64_mod
            orig_b64 = f"data:image/jpeg;base64,{b64_mod.b64encode(image_bytes).decode('ascii')}"
            ann_b64 = loc.get("annotated_image") or orig_b64

            return {
                "module_id": self.module_id,
                "display_name": self.display_name,
                "status": "success" if result.get("status") in ["classified", "review_required"] else result.get("status"),
                "decision": "REJECT (DEFECT)" if is_defect else ("PERLU REVIEW" if pred_label == "Perlu Review" else "PASS (OK)"),
                "is_defect": is_defect,
                "predicted_label": pred_label,
                "confidence": round(conf, 2),
                "probabilities": {
                    k: round(v * 100.0, 2) for k, v in result.get("scores", {}).items()
                },
                "processing_ms": result.get("processing_ms", 0),
                "quality_details": result.get("quality_details", {}),
                "review_reasons": result.get("review_reasons", []),
                "num_defect_markers": len(boxes),
                "defect_boxes": boxes,
                "original_image": orig_b64,
                "annotated_image": ann_b64,
                "anomaly_matrix": loc.get("anomaly_matrix_image"),
                "metadata": {
                    "model_version": result.get("model_version"),
                    "inspection_id": result.get("inspection_id")
                }
            }
        except Exception as e:
            return {
                "module_id": self.module_id,
                "status": "error",
                "error_message": f"Error inferensi casting: {str(e)}"
            }
