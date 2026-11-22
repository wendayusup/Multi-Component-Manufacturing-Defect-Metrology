"""
UNIFIED API ROUTES
------------------
Versioned API conforming strictly to integration.md specifications.
Supports /api and /api/v1 endpoints for module discovery, single-module
inspection, and cross-module inspection (Run All).
"""
import io
import time
import base64
import random
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file
from ..services.model_registry import get_model_registry
from ..config import WORKSPACE_ROOT, RING_DIR, CASTING_DIR

api_bp = Blueprint("api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# 1. System Health & Discovery Endpoints
# ---------------------------------------------------------------------------

@api_bp.route("/health", methods=["GET"])
@api_bp.route("/v1/health", methods=["GET"])
def get_system_health():
    """
    Returns application-level health and overall module readiness status.
    Conforms to integration.md B9.5.
    """
    registry = get_model_registry()
    modules_health = registry.get_all_health()
    all_ready = all(m.get("status") == "ready" for m in modules_health.values())

    return jsonify({
        "status": "healthy",
        "service": "Unified Industrial Visual Inspection Platform",
        "all_models_ready": all_ready,
        "modules": modules_health,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }), 200


@api_bp.route("/modules", methods=["GET"])
@api_bp.route("/v1/modules", methods=["GET"])
def get_modules():
    """
    Returns discovery metadata and readiness status for all 3 inspection modules.
    Conforms to integration.md B9.1.
    """
    registry = get_model_registry()
    return jsonify({
        "status": "success",
        "modules": registry.get_all_health()
    }), 200


@api_bp.route("/modules/<module_id>/health", methods=["GET"])
@api_bp.route("/v1/modules/<module_id>/health", methods=["GET"])
def get_single_module_health(module_id):
    """
    Returns health status and metadata for a specific module.
    Conforms to integration.md B9.2.
    """
    registry = get_model_registry()
    adapter = registry.get_adapter(module_id)
    if not adapter:
        return jsonify({
            "error_code": "MODULE_NOT_FOUND",
            "message": f"Module '{module_id}' not found. Available modules: ring, bearing, plastic."
        }), 404

    return jsonify(adapter.health()), 200


# ---------------------------------------------------------------------------
# 2. Image Decoding Helper
# ---------------------------------------------------------------------------

def extract_image_bytes_from_request():
    """Extracts raw image bytes and original filename from multipart or JSON payload."""
    image_bytes = None
    filename = "input.jpg"

    # Multipart file upload
    file = request.files.get("file") or request.files.get("image")
    if file and file.filename != "":
        filename = file.filename
        image_bytes = file.read()

    # JSON base64 payload (camera snapshots)
    elif request.is_json and "image_base64" in request.json:
        data_url = request.json["image_base64"]
        try:
            if "," in data_url:
                header, encoded = data_url.split(",", 1)
            else:
                encoded = data_url
            image_bytes = base64.b64decode(encoded)
        except Exception as e:
            return None, filename, f"Invalid base64 payload: {str(e)}"

    return image_bytes, filename, None


# ---------------------------------------------------------------------------
# 3. Single-Module Inspection Endpoint
# ---------------------------------------------------------------------------

@api_bp.route("/inspect/<module_id>", methods=["POST"])
@api_bp.route("/v1/inspect/<module_id>", methods=["POST"])
def inspect_module(module_id):
    """
    Unified single-module inspection endpoint.
    Conforms to integration.md B9.3.
    """
    registry = get_model_registry()
    adapter = registry.get_adapter(module_id)

    if not adapter:
        return jsonify({
            "status": "invalid_input",
            "error_code": "MODULE_NOT_FOUND",
            "error_message": f"Module '{module_id}' not found. Available: ring, bearing, plastic."
        }), 404

    image_bytes, filename, decode_err = extract_image_bytes_from_request()
    if decode_err:
        return jsonify({"status": "invalid_input", "error_message": decode_err}), 400
    if not image_bytes:
        return jsonify({
            "status": "invalid_input",
            "error_message": "No image data provided. Upload a file or capture camera frame."
        }), 400

    # Extract module parameters
    kwargs = {"filename": filename}
    if request.is_json:
        if "sensitivity" in request.json:
            kwargs["sensitivity"] = request.json["sensitivity"]
        if "grid_dim" in request.json:
            try:
                kwargs["grid_dim"] = int(request.json["grid_dim"])
            except ValueError:
                pass
    else:
        if "sensitivity" in request.form:
            kwargs["sensitivity"] = request.form["sensitivity"]
        if "grid_dim" in request.form:
            try:
                kwargs["grid_dim"] = int(request.form["grid_dim"])
            except ValueError:
                pass

    # Run inspection
    result = adapter.inspect(image_bytes, **kwargs)
    return jsonify(result)


# ---------------------------------------------------------------------------
# 4. Cross-Module Inspection Endpoint (Run All)
# ---------------------------------------------------------------------------

@api_bp.route("/inspect/all", methods=["POST"])
@api_bp.route("/v1/inspect/all", methods=["POST"])
def inspect_all_modules():
    """
    Runs all available modules against the same source image content.
    Supports partial success when one model is unavailable.
    Conforms to integration.md B7 & B9.4.
    """
    start_time = time.perf_counter()
    image_bytes, filename, decode_err = extract_image_bytes_from_request()
    if decode_err:
        return jsonify({"status": "invalid_input", "error_message": decode_err}), 400
    if not image_bytes:
        return jsonify({"status": "invalid_input", "error_message": "No image provided for Run All."}), 400

    registry = get_model_registry()
    results = {}
    has_success = False
    has_failure = False

    # Execute sequentially to avoid memory spikes (B7 rule)
    for mod_id in ["bearing", "ring", "plastic"]:
        adapter = registry.get_adapter(mod_id)
        if adapter and adapter.is_ready:
            try:
                res = adapter.inspect(image_bytes, filename=filename)
                results[mod_id] = res
                if res.get("status") == "success":
                    has_success = True
                else:
                    has_failure = True
            except Exception as e:
                results[mod_id] = {"status": "error", "error_message": str(e)}
                has_failure = True
        else:
            health = adapter.health() if adapter else {}
            results[mod_id] = {
                "status": "model_unavailable",
                "message": health.get("message") or "Model tidak tersedia."
            }
            has_failure = True

    total_ms = round((time.perf_counter() - start_time) * 1000, 2)
    overall_status = "completed" if (has_success and not has_failure) else ("partial_success" if has_success else "failed")

    return jsonify({
        "status": overall_status,
        "results": results,
        "timing": {"total_ms": total_ms},
        "disclaimer": "Hasil inspeksi AI merupakan analisis awal dan wajib diverifikasi oleh personel QC berkualifikasi."
    })


# ---------------------------------------------------------------------------
# 5. Quick Sample Presets & Academic Benchmarks
# ---------------------------------------------------------------------------

@api_bp.route("/sample/<module_id>/<sample_type>", methods=["GET"])
@api_bp.route("/sample/<sample_type>", methods=["GET"])
def get_sample_image(sample_type, module_id="ring"):
    """
    Serves sample test images for quick operational verification.
    Supports both /api/sample/<module_id>/<sample_type> and legacy /api/sample/<sample_type>.
    """
    # If called with legacy /sample/<sample_type>, module_id might be "ring" by default,
    # or if the first arg matches a module name
    if module_id not in ["ring", "bearing", "plastic"] and sample_type in ["ring", "bearing", "plastic"]:
        module_id, sample_type = sample_type, module_id

    sample_type_clean = (sample_type or "").lower()

    if module_id == "ring":
        subfolder = "defect_ring" if sample_type_clean in ["defect", "defective"] else "normal_ring"
        possible_folders = [
            RING_DIR / "dataset_split" / "test" / subfolder,
            RING_DIR / "dataset_split" / "val" / subfolder,
            RING_DIR / "dataset_split" / "train" / subfolder,
            WORKSPACE_ROOT / "RingClassification" / "dataset_split" / "test" / subfolder,
            Path.cwd() / "model_development" / "ring_inspection" / "dataset_split" / "test" / subfolder,
            Path.cwd() / "Unified_Inspection" / "model_development" / "ring_inspection" / "dataset_split" / "test" / subfolder,
        ]
        for folder in possible_folders:
            if folder.exists():
                files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
                if files:
                    selected = random.choice(files)
                    resp = send_file(str(selected), mimetype="image/jpeg")
                    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                    resp.headers["Pragma"] = "no-cache"
                    resp.headers["Expires"] = "0"
                    return resp

    elif module_id == "bearing":
        subfolder = "def_front" if sample_type_clean in ["defect", "defective"] else "ok_front"
        possible_folders = [
            CASTING_DIR / "casting_512x512" / "casting_512x512" / subfolder,
            CASTING_DIR / "casting_data" / "test" / subfolder,
            WORKSPACE_ROOT / "Casting_Detection" / "casting_512x512" / "casting_512x512" / subfolder,
            Path.cwd() / "model_development" / "bearing_inspection" / "casting_512x512" / "casting_512x512" / subfolder,
            Path.cwd() / "Unified_Inspection" / "model_development" / "bearing_inspection" / "casting_512x512" / "casting_512x512" / subfolder,
        ]
        for folder in possible_folders:
            if folder.exists():
                files = list(folder.glob("*.jpeg")) + list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
                if files:
                    selected = random.choice(files)
                    resp = send_file(str(selected), mimetype="image/jpeg")
                    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                    resp.headers["Pragma"] = "no-cache"
                    resp.headers["Expires"] = "0"
                    return resp

    return jsonify({
        "error": f"No sample found for module '{module_id}' and type '{sample_type}'",
        "hint": "Pastikan dataset_split tersedia pada model_development/ring_inspection/dataset_split/test"
    }), 404


@api_bp.route("/analytics/ring/<analytic_type>", methods=["GET"])
def get_ring_analytics(analytic_type):
    """Serves Ring evaluation metrics and academic figures for S2 defense."""
    results_dir = RING_DIR / "results"
    if analytic_type == "confusion_matrix":
        p = results_dir / "04_confusion_matrix.png"
        if p.exists():
            return send_file(str(p), mimetype="image/png")
    elif analytic_type == "training_history":
        p = results_dir / "03_training_history.png"
        if p.exists():
            return send_file(str(p), mimetype="image/png")
    elif analytic_type == "s2_report":
        return jsonify({
            "status": "success",
            "model_architecture": "Pure 7-Stage Deep CNN (Custom Layer-by-Layer PyTorch)",
            "test_accuracy": "95.83%",
            "defect_precision": "100.00%",
            "defect_recall": "91.67%",
            "f1_score": "0.9565",
            "preprocessing": "512x512 CLAHE Grayscale",
            "training_specs": {
                "epochs": 20,
                "optimizer": "Adam (lr=1e-3, cosine decay)",
                "batch_size": 16,
                "dataset_samples": 48
            },
            "metrology": {
                "matrix_resolution": "512x512 (262,144 analytical cells)",
                "clustering_algorithm": "Deterministic Anomaly Density Clustering (No-Blur Bounding Box)",
                "disparity_calculation": "Residual subtraction against normal baseline"
            }
        })

    return jsonify({"error": "Analytic resource not found"}), 404
