"""
Inference Orchestration Service for Casting Defect Inspection.
Coordinates file decoding, quality gates, preprocessing, CNN prediction,
decision thresholds, uncertainty handling ("Perlu Review"), and audit logging.
"""
import time
import uuid
from typing import Dict, Optional
import numpy as np

import config
from app.image_io import decode_image_bytes
from app.quality_check import check_image_quality
from app.preprocessing import preprocess_pil_image
from app.model_loader import get_model_manager
from app.storage import save_inspection
from cnn.localization import locate_defects, generate_annotated_image_base64, compute_anomaly_matrix_base64


DISCLAIMER_TEXT = (
    "Alat visual screening eksperimental berbasis AI untuk pendukung keputusan awal. "
    "Hasil bukan jaminan sertifikasi mutu dan wajib diverifikasi oleh personel/operator QC yang berkualifikasi."
)


def run_casting_inspection(
    image_bytes: bytes,
    original_filename: str = "upload.jpg",
    sensitivity: str = "standard"
) -> Dict:
    """
    Executes end-to-end inspection pipeline for an uploaded casting product image.
    Returns structured inspection dictionary.
    """
    start_time = time.perf_counter()
    inspection_id = str(uuid.uuid4())
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    warnings = [DISCLAIMER_TEXT]
    review_reasons = []

    # Sensitivity mode → thresholds
    SENSITIVITY_MAP = {
        "standard": {"confidence": 0.70, "threshold_ratio": 0.65},
        "high":     {"confidence": 0.60, "threshold_ratio": 0.55},
        "ultra":    {"confidence": 0.50, "threshold_ratio": 0.45},
    }
    sens = SENSITIVITY_MAP.get(sensitivity, SENSITIVITY_MAP["standard"])
    confidence_gate = sens["confidence"]
    loc_threshold_ratio = sens["threshold_ratio"]

    # 1. Decode Image
    try:
        image = decode_image_bytes(image_bytes)
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "inspection_id": inspection_id,
            "created_at": created_at,
            "module": "casting_defect",
            "component_type": "casting",
            "status": "invalid_input",
            "predicted_label": None,
            "confidence_score": None,
            "scores": {},
            "model_version": config.MODEL_VERSION,
            "quality_status": "failed",
            "quality_details": {},
            "processing_ms": elapsed_ms,
            "review_reasons": [str(e)],
            "warnings": warnings,
            "error_message": str(e)
        }

    # 2. Quality Check
    quality_result = check_image_quality(image)
    if quality_result["warnings"]:
        review_reasons.extend(quality_result["warnings"])

    # Hard quality failure (e.g. below minimum resolution)
    if quality_result["status"] == "failed":
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        result = {
            "inspection_id": inspection_id,
            "created_at": created_at,
            "module": "casting_defect",
            "component_type": "casting",
            "status": "invalid_input",
            "predicted_label": None,
            "confidence_score": None,
            "scores": {},
            "model_version": config.MODEL_VERSION,
            "quality_status": quality_result["status"],
            "quality_details": {
                "dimensions": quality_result["dimensions"],
                "mean_brightness": quality_result["mean_brightness"],
                "blur_score": quality_result["blur_score"]
            },
            "processing_ms": elapsed_ms,
            "review_reasons": review_reasons,
            "warnings": warnings,
            "error_message": "; ".join(quality_result["warnings"])
        }
        return result

    # 3. Verify Model Availability
    model_mgr = get_model_manager()
    if not model_mgr.is_loaded:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        result = {
            "inspection_id": inspection_id,
            "created_at": created_at,
            "module": "casting_defect",
            "component_type": "casting",
            "status": "model_unavailable",
            "predicted_label": None,
            "confidence_score": None,
            "scores": {},
            "model_version": config.MODEL_VERSION,
            "quality_status": quality_result["status"],
            "quality_details": {
                "dimensions": quality_result["dimensions"],
                "mean_brightness": quality_result["mean_brightness"],
                "blur_score": quality_result["blur_score"]
            },
            "processing_ms": elapsed_ms,
            "review_reasons": [model_mgr.load_error or "Model bobot belum dikonfigurasi."],
            "warnings": warnings,
            "error_message": model_mgr.load_error or "Model belum tersedia."
        }
        return result

    # 4. Image Preprocessing
    tensor = preprocess_pil_image(image)

    # 5. Model Inference
    try:
        probs = model_mgr.predict(tensor)[0]
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "inspection_id": inspection_id,
            "created_at": created_at,
            "module": "casting_defect",
            "component_type": "casting",
            "status": "error",
            "predicted_label": None,
            "confidence_score": None,
            "scores": {},
            "model_version": config.MODEL_VERSION,
            "quality_status": quality_result["status"],
            "quality_details": {},
            "processing_ms": elapsed_ms,
            "review_reasons": [f"Kesalahan internal saat inferensi: {str(e)}"],
            "warnings": warnings,
            "error_message": str(e)
        }

    # 6. Score Mapping & Decision Policy
    class_names = config.CLASS_NAMES
    scores_dict = {class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))}
    pred_idx = int(np.argmax(probs))
    raw_label = class_names[pred_idx]
    confidence_score = round(float(probs[pred_idx]), 4)

    # Uncertainty / Decision Gate
    is_uncertain = confidence_score < confidence_gate
    has_quality_warning = quality_result["status"] == "review_recommended"

    if is_uncertain:
        review_reasons.append(
            f"Skor model ({confidence_score * 100:.1f}%) berada di bawah ambang keyakinan minimum "
            f"({config.CONFIDENCE_THRESHOLD * 100:.0f}%). Keputusan ditahan untuk tinjauan operator."
        )

    if is_uncertain or has_quality_warning:
        status = "review_required"
        predicted_label = "Perlu Review"
    else:
        status = "classified"
        predicted_label = raw_label

    # 7. Defect Localization (Bounding Boxes & Marker Dots) + Anomaly Matrix
    defect_localization = {
        "has_defect": False,
        "defect_count": 0,
        "boxes": [],
        "points": [],
        "annotated_image": None,
        "anomaly_matrix_image": None
    }

    if model_mgr.model is not None:
        # Always compute the anomaly matrix (shows near-zero/blue for OK parts)
        try:
            anomaly_b64 = compute_anomaly_matrix_base64(model_mgr.model, image)
            defect_localization["anomaly_matrix_image"] = anomaly_b64
        except Exception as am_err:
            defect_localization["anomaly_matrix_error"] = str(am_err)

        # Localization only for Defective predictions above threshold
        if raw_label == "Defective":
            try:
                loc = locate_defects(model_mgr.model, image, threshold_ratio=loc_threshold_ratio)
                if loc.get("has_defect"):
                    annotated_b64 = generate_annotated_image_base64(image, loc)
                    defect_localization.update({
                        "has_defect": True,
                        "defect_count": loc["defect_count"],
                        "boxes": loc["boxes"],
                        "points": loc["points"],
                        "annotated_image": annotated_b64
                    })
            except Exception as loc_err:
                defect_localization["error"] = str(loc_err)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    result = {
        "inspection_id": inspection_id,
        "created_at": created_at,
        "module": "casting_defect",
        "component_type": "casting",
        "status": status,
        "predicted_label": predicted_label,
        "confidence_score": confidence_score,
        "scores": scores_dict,
        "model_version": model_mgr.metadata.get("model_version", config.MODEL_VERSION),
        "quality_status": quality_result["status"],
        "quality_details": {
            "dimensions": quality_result["dimensions"],
            "mean_brightness": quality_result["mean_brightness"],
            "blur_score": quality_result["blur_score"]
        },
        "defect_localization": defect_localization,
        "processing_ms": elapsed_ms,
        "review_reasons": review_reasons,
        "warnings": warnings
    }

    # Save to history storage
    try:
        save_inspection(result)
    except Exception as e:
        # Non-fatal logging if DB write fails
        pass

    return result

