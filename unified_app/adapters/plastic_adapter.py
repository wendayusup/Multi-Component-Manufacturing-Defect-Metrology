"""
PLASTIC INJECTION DEFECT INSPECTION ADAPTER
-------------------------------------------
Integrates the Plastic Defect Detection module (PlasticDefectionWeb):
- Single-Class Feature Pyramid CNN with Center & Geometry Heads
- Weakly-supervised / Center-based defect bounding box decoding & NMS
- Dynamic scale-preserved input sizing (768x768 canvas padding)
- Deterministic threshold gating from Colab validation checkpoint
"""
import sys
import io
import time
import base64
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageOps

# Ensure plastic_inspection root is in sys.path
UNIFIED_ROOT = Path(__file__).resolve().parent.parent.parent
PLASTIC_DIR = UNIFIED_ROOT / "model_development" / "plastic_inspection"
if not PLASTIC_DIR.exists():
    PLASTIC_DIR = UNIFIED_ROOT.parent / "PlasticDefectionWeb"
if str(PLASTIC_DIR) not in sys.path:
    sys.path.insert(0, str(PLASTIC_DIR))

from .base import BaseInspectionAdapter


def pil_to_base64(pil_img: Image.Image, format: str = "JPEG", quality: int = 92) -> str:
    """Converts PIL image into a Base64 data URL string."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format, quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


class PlasticAdapter(BaseInspectionAdapter):
    module_id = "plastic"
    display_name = "Plastic Defect Inspection"
    category = "Plastic Injection"

    def __init__(self):
        self.bundle: Optional[Tuple[Any, int, float]] = None
        self.is_ready: bool = False
        self.load_error: Optional[str] = None
        self.side: int = 768
        self.threshold: float = 0.278
        self.load()

    def load(self) -> None:
        """Loads and caches the PyTorch PlasticDetector model bundle."""
        try:
            # Locate model checkpoint file
            model_path = PLASTIC_DIR / "models" / "plastic_defect_detector.pt"
            if not model_path.exists():
                fallback_path = UNIFIED_ROOT.parent / "PlasticDefectionWeb" / "models" / "plastic_defect_detector.pt"
                if fallback_path.exists():
                    model_path = fallback_path
                else:
                    self.is_ready = False
                    self.load_error = f"Checkpoint tidak ditemukan pada: {model_path}"
                    return

            # Dynamic import of detector module to prevent tight coupling
            detector_mod = importlib.import_module("models.detector")
            model, side, threshold = detector_mod.load_detector(model_path, device="cpu")
            self.bundle = (model, side, threshold)
            self.side = side
            self.threshold = float(threshold)
            self.is_ready = True
            self.load_error = None
        except Exception as e:
            self.is_ready = False
            self.load_error = f"Gagal memuat model deteksi plastik: {str(e)}"

    def health(self) -> Dict[str, Any]:
        """Reports module readiness, model metadata, and health details."""
        return {
            "module_id": self.module_id,
            "display_name": self.display_name,
            "category": self.category,
            "status": "ready" if self.is_ready else "error",
            "model_version": "plastic-cnn-fpn-v1.0",
            "input_resolution": f"{self.side}x{self.side} Canvas Padding (Single-Class CNN)",
            "detection_threshold": round(self.threshold, 4),
            "error_message": self.load_error,
            "capabilities": ["object_detection", "defect_localization", "bounding_box_nms", "defect_counting"]
        }

    def inspect(self, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """
        Runs inference on uploaded plastic product photo:
        1. Decodes and transposes EXIF orientation.
        2. Feeds image through PlasticDetector with reflection padding and NMS.
        3. Returns unified standardized QC results with bounding boxes and confidence.
        """
        if not self.is_ready or self.bundle is None:
            return {
                "module_id": self.module_id,
                "status": "model_unavailable",
                "decision": "STANDBY",
                "predicted_label": "Modul Belum Siap",
                "confidence": 0.0,
                "error_message": self.load_error or "Model Plastic belum siap."
            }

        start_time = time.perf_counter()
        try:
            # 1. Open image from raw bytes
            with Image.open(io.BytesIO(image_bytes)) as raw_img:
                image = ImageOps.exif_transpose(raw_img).convert("RGB")
                orig_w, orig_h = image.size

            # 2. Run inference via detector module
            detector_mod = importlib.import_module("models.detector")
            annotated_img, detections = detector_mod.inspect_photo(image, self.bundle)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # 3. Analyze detections
            num_defects = len(detections)
            is_defect = (num_defects > 0)

            if is_defect:
                max_conf = max(float(d.get("confidence", 0.0)) for d in detections)
                conf = round(max_conf * 100.0, 2)
                decision = "REJECT (DEFECT)"
                pred_label = "Defective"
                probabilities = {
                    "Defective": conf,
                    "OK": round(max(0.0, 100.0 - conf), 2)
                }
            else:
                conf = round(max(85.0, (1.0 - self.threshold) * 100.0), 2)
                decision = "PASS (OK)"
                pred_label = "OK"
                probabilities = {
                    "Defective": round(max(0.0, 100.0 - conf), 2),
                    "OK": conf
                }

            # 4. Standardize boxes list for UI rendering
            standardized_boxes = []
            for i, d in enumerate(detections, start=1):
                raw_xyxy = d.get("xyxy", [0, 0, 0, 0])
                box_conf = float(d.get("confidence", 0.0))
                standardized_boxes.append({
                    "id": i,
                    "label": "Cacat Injeksi",
                    "confidence": round(box_conf * 100.0, 2),
                    "xyxy": [round(coord, 1) for coord in raw_xyxy],
                    "box_int": [int(round(coord)) for coord in raw_xyxy]
                })

            # 5. Base64 encode images
            orig_b64 = pil_to_base64(image)
            ann_b64 = pil_to_base64(annotated_img)

            return {
                "module_id": self.module_id,
                "display_name": self.display_name,
                "status": "success",
                "decision": decision,
                "is_defect": is_defect,
                "predicted_label": pred_label,
                "confidence": conf,
                "probabilities": probabilities,
                "processing_ms": elapsed_ms,
                "num_defect_markers": num_defects,
                "defect_boxes": standardized_boxes,
                "detection_threshold": round(self.threshold, 4),
                "image_size": {"width": orig_w, "height": orig_h},
                "original_image": orig_b64,
                "annotated_image": ann_b64,
                "metadata": {
                    "side": self.side,
                    "threshold": self.threshold,
                    "model_name": "PlasticDetector Single-Class CNN"
                }
            }
        except Exception as e:
            return {
                "module_id": self.module_id,
                "status": "error",
                "error_message": f"Kesalahan inferensi modul plastik: {str(e)}"
            }
