"""
BEARING RING CLASSIFICATION & METROLOGY ADAPTER
-----------------------------------------------
Integrates the Bearing Ring Defect Inspection module (RingClassification folder):
- Pure 7-Stage CNN trained on 512x512 CLAHE dataset (585,250 weights)
- Deterministic cluster-based defect bounding boxes
- High-definition 512x512 Metrology Disparity Matrix (262,144 cells)
- Multi-layer industrial filters: CLAHE, Black-Hat morphology, Sobel texture gradient
"""
import sys
import io
import time
import base64
import importlib
from pathlib import Path
from typing import Dict, Any
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure ring_inspection root is accessible
UNIFIED_ROOT = Path(__file__).resolve().parent.parent.parent
RING_DIR = UNIFIED_ROOT / "model_development" / "ring_inspection"
if not RING_DIR.exists():
    RING_DIR = UNIFIED_ROOT.parent / "RingClassification"
if str(RING_DIR) not in sys.path:
    sys.path.insert(0, str(RING_DIR))

from .base import BaseInspectionAdapter


def pil_to_base64(pil_img, format="JPEG"):
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format, quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


class RingAdapter(BaseInspectionAdapter):
    module_id = "ring"
    display_name = "Bearing Ring Inspection"
    category = "Bearing Surface"

    def __init__(self):
        self.localizer = None
        self.is_ready = False
        self.load_error = None
        self.load()

    def load(self) -> None:
        try:
            defect_localizer_mod = importlib.import_module("step05_inference.defect_localizer")
            self.localizer = defect_localizer_mod.DefectLocalizer()
            self.is_ready = True
        except Exception as e:
            self.is_ready = False
            self.load_error = str(e)

    def health(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "display_name": self.display_name,
            "category": self.category,
            "status": "ready" if self.is_ready else "error",
            "model_architecture": "Pure 7-Stage CNN + Metrology Safety Interlock",
            "input_resolution": "512x512 CLAHE Grayscale",
            "accuracy": "95.83% (Precision: 100.0%)",
            "error_message": self.load_error,
            "capabilities": [
                "classification",
                "dual_gate_safety_interlock",
                "cluster_bounding_boxes",
                "metrology_matrix_512",
                "filter_clahe",
                "filter_blackhat",
                "filter_sobel"
            ]
        }

    def inspect(self, image_bytes: bytes, grid_dim: int = 512, sensitivity: float = 0.25, **kwargs) -> Dict[str, Any]:
        if not self.is_ready or self.localizer is None:
            return {
                "module_id": self.module_id,
                "status": "model_unavailable",
                "error_message": self.load_error or "Model Ring belum siap."
            }

        try:
            start_time = time.perf_counter()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            orig_np = np.array(pil_img)
            orig_hw = (orig_np.shape[0], orig_np.shape[1])
            h, w = orig_hw

            if grid_dim not in (128, 256, 512, 1024, 2048, 4096):
                grid_dim = 512

            # 1. CLAHE & Ring Metal Mask
            gray_raw = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_clahe = clahe_obj.apply(gray_raw)
            ring_mask = self.localizer._extract_ring_mask(gray_clahe, orig_hw)

            # 2. Gate 1: Pure 7-Stage Deep CNN
            from step05_inference.defect_localizer import INFERENCE_TRANSFORM, CLASS_NAMES, INDUSTRIAL_FILTER_METADATA  # type: ignore
            tensor_img = INFERENCE_TRANSFORM(pil_img).unsqueeze(0).to(self.localizer.device)
            with torch.no_grad():
                output = self.localizer.model(tensor_img)
                probabilities = F.softmax(output, dim=1).cpu().numpy()[0]

            cnn_pred_idx = int(probabilities.argmax())
            cnn_pred_label = CLASS_NAMES[cnn_pred_idx]
            cnn_conf = float(probabilities[cnn_pred_idx]) * 100.0
            cnn_is_defect = (cnn_pred_label == "defect_ring")

            # 3. Gate 2: Surface Metrology & Texture Roughness (Sobel Topography + Black-Hat)
            blurred = cv2.GaussianBlur(gray_raw, (7, 7), 0)
            sx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(sx, sy)

            gray_res = cv2.resize(gray_clahe, (grid_dim, grid_dim)).astype(np.float32) / 255.0
            mask_res = cv2.resize(ring_mask, (grid_dim, grid_dim), interpolation=cv2.INTER_NEAREST) > 0.5
            k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
            mask_inner = cv2.erode(mask_res.astype(np.uint8), k_erode) > 0

            sobel_mean = float(np.mean(cv2.resize(grad_mag, (grid_dim, grid_dim))[mask_inner])) if mask_inner.any() else 0.0

            k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            bh = (cv2.morphologyEx(gray_res, cv2.MORPH_BLACKHAT, k5) * 0.60) + (cv2.morphologyEx(gray_res, cv2.MORPH_BLACKHAT, k9) * 0.40)
            bh_ring = bh * mask_inner
            vals = bh_ring[mask_inner] if mask_inner.any() else np.array([0.0])
            bh_p95 = float(np.percentile(vals, 95)) if len(vals) > 0 else 0.0

            # Metrology Safety Interlock trigger:
            # Baseline normal: sobel_mean in [9.9, 21.1], bh_p95 in [0.18, 0.34]
            # Real defects (including smartphone close-up/OOD): sobel_mean in [33.0, 46.0]
            metrology_trigger = (sobel_mean >= 26.0)

            effective_is_defect = cnn_is_defect or metrology_trigger

            # 4. Compute Dynamic Disparity Matrix & Cluster Bounding Boxes
            matrix_vals, boxes = self.localizer._compute_matrix_and_bounding_boxes(
                gray_clahe, orig_hw, ring_mask, is_defect=effective_is_defect, grid_dim=grid_dim
            )

            # Final Dual-Gate Decision
            final_is_defect = effective_is_defect and (len(boxes) > 0 or cnn_is_defect)
            decision = "REJECT (DEFECT)" if final_is_defect else "PASS (NORMAL)"
            predicted_class = "defect_ring" if final_is_defect else "normal_ring"

            if cnn_is_defect:
                gate_info = "Gate 1 (Pure CNN)"
                confidence = cnn_conf
                final_probs = {
                    CLASS_NAMES[i]: float(probabilities[i]) * 100.0 for i in range(2)
                }
            elif final_is_defect:
                gate_info = "Gate 2 (Metrology Interlock)"
                confidence = min(98.5, max(88.0, sobel_mean * 2.15))
                final_probs = {
                    "defect_ring": round(confidence, 2),
                    "normal_ring": round(100.0 - confidence, 2)
                }
            else:
                gate_info = "Gate 1 + 2 (Clear)"
                confidence = cnn_conf
                final_probs = {
                    CLASS_NAMES[i]: float(probabilities[i]) * 100.0 for i in range(2)
                }

            # 5. Render Visual Views
            annotated_image = self.localizer._draw_boxes_on_image(orig_np, boxes, is_defect=final_is_defect)
            matrix_grid_image = self.localizer._render_matrix_grid_view(matrix_vals, grid_dim=grid_dim, out_size=1024)
            industrial_filters = self.localizer._apply_industrial_filters(orig_np, gray_clahe=gray_clahe)

            views = {
                "red_dots": annotated_image,
                "chessboard_matrix": matrix_grid_image,
                "original_native": pil_img
            }
            views.update(industrial_filters)

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            orig_b64 = pil_to_base64(pil_img)
            annotated_b64 = pil_to_base64(annotated_image)
            views_b64 = {name: pil_to_base64(img) for name, img in views.items()}

            metrics = {
                "matrix_shape": f"{matrix_vals.shape[0]}x{matrix_vals.shape[1]}",
                "matrix_total_cells": int(matrix_vals.size),
                "grid_dim": grid_dim,
                "matrix_max": int(np.max(matrix_vals)),
                "matrix_mean": round(float(np.mean(matrix_vals[matrix_vals > 0])), 2) if np.sum(matrix_vals > 0) > 0 else 0.0,
                "anomaly_boxes_count": len(boxes),
                "defect_probability": round(float(final_probs["defect_ring"]), 2),
                "normal_probability": round(float(final_probs["normal_ring"]), 2),
                "sobel_surface_gradient": round(sobel_mean, 2),
                "blackhat_p95_disparity": round(bh_p95, 4),
                "gate_triggered": gate_info
            }

            return {
                "module_id": self.module_id,
                "display_name": self.display_name,
                "status": "success",
                "decision": decision,
                "is_defect": final_is_defect,
                "predicted_label": predicted_class,
                "confidence": round(confidence, 2),
                "probabilities": {
                    k: round(v, 2) for k, v in final_probs.items()
                },
                "processing_ms": latency_ms,
                "num_defect_markers": len(boxes),
                "gate_info": gate_info,
                "original_image": orig_b64,
                "annotated_image": annotated_b64,
                "views": views_b64,
                "metrics": metrics,
                "filter_metadata": INDUSTRIAL_FILTER_METADATA
            }
        except Exception as e:
            return {
                "module_id": self.module_id,
                "status": "error",
                "error_message": f"Error inferensi ring: {str(e)}"
            }
