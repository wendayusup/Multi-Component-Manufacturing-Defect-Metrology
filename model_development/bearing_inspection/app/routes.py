"""
Flask Routes and API Endpoints for Casting Defect Detector.
Provides web page rendering, casting inspection REST API, history retrieval,
and operator review updates.
"""
import os
import random
from flask import Blueprint, render_template, request, jsonify, current_app

import config
from app.validation import validate_upload
from app.inference_service import run_casting_inspection
from app.storage import get_recent_inspections, update_manual_review
from app.model_loader import get_model_manager

bp = Blueprint("casting", __name__)


@bp.route("/", methods=["GET"])
def index():
    """Renders the single-module Casting Defect Inspection web interface."""
    model_mgr = get_model_manager()
    return render_template(
        "index.html",
        model_version=config.MODEL_VERSION,
        model_is_loaded=model_mgr.is_loaded,
        model_error=model_mgr.load_error,
        max_size_mb=int(config.MAX_CONTENT_LENGTH / (1024 * 1024)),
        confidence_threshold=int(config.CONFIDENCE_THRESHOLD * 100)
    )


@bp.route("/api/inspect/casting", methods=["POST"])
def inspect_casting():
    """
    Main inspection endpoint.
    Accepts multipart/form-data with a single 'image' file field.
    Returns structured inspection JSON.
    """
    if "image" not in request.files:
        return jsonify({
            "status": "invalid_input",
            "module": "casting_defect",
            "error_message": "Field 'image' tidak ditemukan dalam form upload.",
            "warnings": ["Unggah satu file gambar (JPG/PNG)."]
        }), 400

    file_obj = request.files["image"]

    try:
        image_bytes, original_filename = validate_upload(file_obj)
    except ValueError as val_err:
        return jsonify({
            "status": "invalid_input",
            "module": "casting_defect",
            "error_message": str(val_err),
            "warnings": [str(val_err)]
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "module": "casting_defect",
            "error_message": f"Terjadi kesalahan saat membaca file: {str(e)}",
            "warnings": ["Harap coba unggah kembali file gambar yang valid."]
        }), 500

    # Run inspection pipeline
    sensitivity = request.form.get("sensitivity", "standard").strip().lower()
    if sensitivity not in ("standard", "high", "ultra"):
        sensitivity = "standard"
    result = run_casting_inspection(image_bytes, original_filename=original_filename, sensitivity=sensitivity)

    status_code = 200
    if result.get("status") == "invalid_input":
        status_code = 400
    elif result.get("status") == "error":
        status_code = 500

    return jsonify(result), status_code


@bp.route("/api/history", methods=["GET"])
def get_history():
    """Retrieves recent inspection history records."""
    limit = request.args.get("limit", default=15, type=int)
    history = get_recent_inspections(limit=limit)
    return jsonify({
        "status": "success",
        "count": len(history),
        "inspections": history
    }), 200


@bp.route("/api/history/export", methods=["GET"])
def export_history_csv():
    """
    Exports recent inspection records as a downloadable CSV report (FR-CAST-017).
    """
    import csv
    import io
    from flask import Response

    limit = request.args.get("limit", default=100, type=int)
    records = get_recent_inspections(limit=limit)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Inspection ID",
        "Created At",
        "Module",
        "Status",
        "Predicted Label",
        "Confidence Score",
        "Model Version",
        "Quality Status",
        "Processing ms",
        "Operator Review",
        "Operator Note"
    ])

    for r in records:
        writer.writerow([
            r.get("inspection_id", ""),
            r.get("created_at", ""),
            r.get("module", ""),
            r.get("status", ""),
            r.get("predicted_label", ""),
            r.get("confidence_score", ""),
            r.get("model_version", ""),
            r.get("quality_status", ""),
            r.get("processing_ms", ""),
            r.get("operator_review", ""),
            r.get("operator_note", "")
        ])

    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=casting_inspections.csv"}
    )


@bp.route("/api/inspections/<inspection_id>/review", methods=["POST"])
def update_review(inspection_id):
    """
    Allows a QC operator to record a manual verification status and notes
    without overwriting the underlying AI prediction.
    """
    data = request.get_json(silent=True) or {}
    operator_review = data.get("operator_review", "").strip()
    operator_note = data.get("operator_note", "").strip()

    if not operator_review:
        return jsonify({
            "status": "error",
            "error_message": "Status verifikasi manual (operator_review) wajib diisi."
        }), 400

    success = update_manual_review(inspection_id, operator_review, operator_note)
    if not success:
        return jsonify({
            "status": "error",
            "error_message": f"Data inspeksi ID '{inspection_id}' tidak ditemukan."
        }), 404

    return jsonify({
        "status": "success",
        "inspection_id": inspection_id,
        "operator_review": operator_review,
        "operator_note": operator_note
    }), 200


@bp.route("/api/model/status", methods=["GET"])
def model_status():
    """Returns current model status and active metadata."""
    model_mgr = get_model_manager()
    return jsonify({
        "module": "casting_defect",
        "is_loaded": model_mgr.is_loaded,
        "model_version": model_mgr.metadata.get("model_version", config.MODEL_VERSION),
        "class_names": config.CLASS_NAMES,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "metadata": model_mgr.metadata,
        "load_error": model_mgr.load_error
    }), 200


@bp.route("/api/model/reload", methods=["POST"])
def reload_model():
    """Reloads the weights and metadata from disk without restarting server."""
    model_mgr = get_model_manager()
    success = model_mgr.reload()
    if success:
        return jsonify({
            "status": "success",
            "message": "Model weights berhasil dimuat ulang.",
            "model_version": model_mgr.metadata.get("model_version", config.MODEL_VERSION),
            "metadata": model_mgr.metadata
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": model_mgr.load_error or "Gagal memuat ulang model."
        }), 500


@bp.route("/api/quicktest/<label>", methods=["GET"])
def quicktest(label):
    """
    Loads a random sample image from the dataset for quick testing.
    label: 'defect' or 'normal'
    Returns the inspection result as JSON.
    """
    import os
    import random

    label_map = {
        "defect": config.DATASET_DIR / "casting_data" / "casting_data" / "test" / "def_front",
        "normal": config.DATASET_DIR / "casting_data" / "casting_data" / "test" / "ok_front",
    }

    folder = label_map.get(label)
    if folder is None or not folder.exists():
        return jsonify({
            "status": "error",
            "error_message": f"Folder dataset untuk label '{label}' tidak ditemukan."
        }), 404

    image_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        return jsonify({
            "status": "error",
            "error_message": "Tidak ada gambar dalam folder dataset."
        }), 404

    sample_file = random.choice(image_files)
    sample_path = folder / sample_file

    with open(sample_path, "rb") as f:
        image_bytes = f.read()

    sensitivity = request.args.get("sensitivity", "standard").strip().lower()
    if sensitivity not in ("standard", "high", "ultra"):
        sensitivity = "standard"

    result = run_casting_inspection(image_bytes, original_filename=sample_file, sensitivity=sensitivity)
    result["quicktest_file"] = sample_file

    # Encode the original image as base64 so the frontend can display it
    import base64
    with open(sample_path, "rb") as f:
        orig_b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = sample_file.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    result["original_image_b64"] = f"data:{mime};base64,{orig_b64}"

    status_code = 200
    if result.get("status") == "invalid_input":
        status_code = 400
    elif result.get("status") == "error":
        status_code = 500

    return jsonify(result), status_code
