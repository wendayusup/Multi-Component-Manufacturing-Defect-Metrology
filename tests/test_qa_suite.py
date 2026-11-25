"""
=============================================================================
UNIFIED INDUSTRIAL INSPECTION PLATFORM - AUTOMATED QA TEST SUITE
=============================================================================
Quality Assurance test suite validating end-to-end functionality, model readiness,
API schema contracts, error handling, and frontend DOM integrity.
Designed to guarantee a stable baseline before further development by the 
Bearing and Plastic inspection teams.
=============================================================================
"""
import io
import os
import sys
import json
import base64
import unittest
from pathlib import Path
from PIL import Image

# Ensure Unified_Inspection is on sys.path
TEST_DIR = Path(__file__).resolve().parent
UNIFIED_ROOT = TEST_DIR.parent
if str(UNIFIED_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIFIED_ROOT))

from unified_app import create_app
from unified_app.services.model_registry import get_model_registry


class TestUnifiedInspectionQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

        # Cache valid samples for subsequent tests
        r_def = cls.client.get("/api/sample/ring/defect")
        cls.ring_defect_bytes = r_def.data if r_def.status_code == 200 else b""

        r_norm = cls.client.get("/api/sample/ring/normal")
        cls.ring_normal_bytes = r_norm.data if r_norm.status_code == 200 else b""

        b_def = cls.client.get("/api/sample/bearing/defect")
        cls.bearing_defect_bytes = b_def.data if b_def.status_code == 200 else b""

        b_norm = cls.client.get("/api/sample/bearing/normal")
        cls.bearing_normal_bytes = b_norm.data if b_norm.status_code == 200 else b""

    # -------------------------------------------------------------------------
    # 1. Health & Discovery Architecture Tests
    # -------------------------------------------------------------------------
    def test_01_system_health(self):
        """Verifies /api/v1/health conforms to schema and reports correct module statuses."""
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("modules", data)
        self.assertEqual(data["modules"]["ring"]["status"], "ready")
        self.assertEqual(data["modules"]["bearing"]["status"], "ready")
        self.assertEqual(data["modules"]["plastic"]["status"], "ready")

    def test_02_module_discovery(self):
        """Verifies /api/v1/modules lists metadata for all 3 modules."""
        res = self.client.get("/api/v1/modules")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("ring", data["modules"])
        self.assertIn("bearing", data["modules"])
        self.assertIn("plastic", data["modules"])

    def test_03_single_module_health(self):
        """Verifies individual health endpoints for ring, bearing, plastic, and 404 for unknown."""
        for mod in ["ring", "bearing", "plastic"]:
            res = self.client.get(f"/api/v1/modules/{mod}/health")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["module_id"], mod)

        res_404 = self.client.get("/api/v1/modules/nonexistent_module/health")
        self.assertEqual(res_404.status_code, 404)

    # -------------------------------------------------------------------------
    # 2. Preset Sample Feeds Tests
    # -------------------------------------------------------------------------
    def test_04_sample_image_feeds(self):
        """Verifies that sample presets load valid JPEG binary images."""
        self.assertGreater(len(self.ring_defect_bytes), 1000)
        self.assertGreater(len(self.ring_normal_bytes), 1000)
        self.assertGreater(len(self.bearing_defect_bytes), 1000)
        self.assertGreater(len(self.bearing_normal_bytes), 1000)

        # Plastic sample feeds should return 200 valid JPEG
        res_p = self.client.get("/api/sample/plastic/defect")
        self.assertEqual(res_p.status_code, 200)
        self.assertEqual(res_p.content_type, "image/jpeg")

        # Unknown sample should return 404 gracefully
        res_unk = self.client.get("/api/sample/plastic/unknown_type")
        self.assertEqual(res_unk.status_code, 404)

    # -------------------------------------------------------------------------
    # 3. Bearing Ring Inspection Tests (Deep 7-Stage CNN)
    # -------------------------------------------------------------------------
    def test_05_ring_inspection_defect_multipart(self):
        """Tests ring inspection with defective sample via multipart upload."""
        res = self.client.post(
            "/api/v1/inspect/ring",
            data={"file": (io.BytesIO(self.ring_defect_bytes), "defect.jpg"), "grid_dim": 512},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("decision", data)
        self.assertIn("views", data)
        self.assertIn("red_dots", data["views"])
        self.assertIn("chessboard_matrix", data["views"])
        self.assertIn("filter_clahe", data["views"])

    def test_06_ring_inspection_normal_json_base64(self):
        """Tests ring inspection with normal sample via JSON base64 payload."""
        b64_str = f"data:image/jpeg;base64,{base64.b64encode(self.ring_normal_bytes).decode('ascii')}"
        res = self.client.post(
            "/api/v1/inspect/ring",
            json={"image_base64": b64_str, "grid_dim": 256}
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertFalse(data["is_defect"])
        self.assertIn("PASS", data["decision"])
        self.assertEqual(data["metrics"]["matrix_shape"], "256x256")

    # -------------------------------------------------------------------------
    # 4. Bearing / Casting Inspection Tests (Pure-NumPy Sequential CNN + Grad-CAM)
    # -------------------------------------------------------------------------
    def test_07_bearing_inspection_defect_multipart(self):
        """Tests bearing/casting inspection with defective sample via multipart."""
        res = self.client.post(
            "/api/v1/inspect/bearing",
            data={"file": (io.BytesIO(self.bearing_defect_bytes), "cast_def.jpg"), "sensitivity": "standard"},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("decision", data)
        self.assertIn("quality_details", data)
        self.assertIn("blur_score", data["quality_details"])
        self.assertIn("mean_brightness", data["quality_details"])

    def test_08_bearing_inspection_sensitivity_policies(self):
        """Tests bearing inspection under High and Ultra sensitivity policies."""
        b64_str = f"data:image/jpeg;base64,{base64.b64encode(self.bearing_defect_bytes).decode('ascii')}"
        for policy in ["high", "ultra"]:
            res = self.client.post(
                "/api/v1/inspect/bearing",
                json={"image_base64": b64_str, "sensitivity": policy}
            )
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["status"], "success")
            self.assertIn("decision", data)

    # -------------------------------------------------------------------------
    # 5. Plastic Inspection Integration Tests
    # -------------------------------------------------------------------------
    def test_09_plastic_inspection_live(self):
        """
        Verifies that calling the plastic inspection endpoint executes successfully
        using the integrated Single-Class FPN PyTorch CNN model.
        """
        res = self.client.post(
            "/api/v1/inspect/plastic",
            data={"file": (io.BytesIO(self.bearing_normal_bytes), "sample.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["module_id"], "plastic")
        self.assertEqual(data["status"], "success")
        self.assertIn(data["decision"], ["PASS (OK)", "REJECT (DEFECT)"])
        self.assertIn("defect_boxes", data)
        self.assertIn("original_image", data)
        self.assertIn("annotated_image", data)

    # -------------------------------------------------------------------------
    # 6. Run All (Cross-Module Inspection) Tests
    # -------------------------------------------------------------------------
    def test_10_inspect_all_modules(self):
        """Verifies /api/v1/inspect/all executes sequentially across all 3 modules."""
        res = self.client.post(
            "/api/v1/inspect/all",
            data={"file": (io.BytesIO(self.ring_defect_bytes), "input.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "completed")
        self.assertIn("bearing", data["results"])
        self.assertIn("ring", data["results"])
        self.assertIn("plastic", data["results"])
        self.assertEqual(data["results"]["bearing"]["status"], "success")
        self.assertEqual(data["results"]["ring"]["status"], "success")
        self.assertEqual(data["results"]["plastic"]["status"], "success")

    # -------------------------------------------------------------------------
    # 7. Error Handling & Validation Tests
    # -------------------------------------------------------------------------
    def test_11_missing_image_payload(self):
        """Verifies HTTP 400 is returned when no image data is sent."""
        res = self.client.post("/api/v1/inspect/ring", json={})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["status"], "invalid_input")

    def test_12_corrupted_base64_payload(self):
        """Verifies HTTP 400 is returned when an invalid base64 string is provided."""
        res = self.client.post(
            "/api/v1/inspect/bearing",
            json={"image_base64": "not_valid_base64_data!!!"}
        )
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["status"], "invalid_input")

    def test_13_corrupted_image_binary(self):
        """Verifies graceful handling when non-image arbitrary bytes are uploaded."""
        res = self.client.post(
            "/api/v1/inspect/ring",
            data={"file": (io.BytesIO(b"THIS_IS_NOT_AN_IMAGE_FILE"), "corrupt.jpg")},
            content_type="multipart/form-data"
        )
        # Should return a controlled failure response without crashing server
        self.assertIn(res.status_code, [200, 400])
        data = res.get_json()
        self.assertIn(data.get("status"), ["error", "invalid_input"])

    # -------------------------------------------------------------------------
    # 8. Academic Benchmark & S2 Analytics Endpoints
    # -------------------------------------------------------------------------
    def test_14_academic_analytics_endpoints(self):
        """Verifies that academic figures and benchmark reports are accessible."""
        res_rep = self.client.get("/api/analytics/ring/s2_report")
        self.assertEqual(res_rep.status_code, 200)
        rep = res_rep.get_json()
        self.assertEqual(rep["test_accuracy"], "95.83%")
        self.assertEqual(rep["defect_precision"], "100.00%")

        res_cm = self.client.get("/api/analytics/ring/confusion_matrix")
        self.assertEqual(res_cm.status_code, 200)
        self.assertEqual(res_cm.content_type, "image/png")

        res_th = self.client.get("/api/analytics/ring/training_history")
        self.assertEqual(res_th.status_code, 200)
        self.assertEqual(res_th.content_type, "image/png")

    # -------------------------------------------------------------------------
    # 9. Plastic Sample Presets Endpoints
    # -------------------------------------------------------------------------
    def test_15_plastic_sample_endpoints(self):
        """Verifies /api/sample/plastic/defect and /normal return valid images."""
        for s_type in ["defect", "normal"]:
            res = self.client.get(f"/api/sample/plastic/{s_type}")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.content_type, "image/jpeg")

    # -------------------------------------------------------------------------
    # 10. UI DOM Integrity & Template Element Mapping Tests
    # -------------------------------------------------------------------------
    def test_16_ui_dom_integrity(self):
        """
        Verifies that the rendered HTML contains all required DOM containers and
        interactive element IDs used by app.js, ring.js, and bearing.js.
        """
        res = self.client.get("/inspection/ring")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        required_ids = [
            # Navbar & Global
            "btnNavRing", "btnNavBearing", "btnNavPlastic",
            "workspaceRing", "workspaceBearing", "workspacePlastic",
            "zoomModal", "zoomImg", "zoomTitle",
            # Ring Module
            "ringTabUpload", "ringTabCam", "ringUploadArea", "ringCameraArea",
            "ringFileInput", "ringVideo", "btnRingAuto", "ringAutoTxt",
            "ringMatrixSlider", "activeResLabel",
            "rImg1", "rPh1", "rImg2", "rPh2", "rCardOutput", "rOutputTag",
            "rDecisionBanner", "rDecisionTxt", "rConfTxt", "rBarDefect", "rBarNormal",
            # Bearing Module (Secondary Module: dropzone & presets)
            "castUploadArea", "castFileInput",
            "sensStd", "sensHigh", "sensUltra",
            "cImg1", "cPh1", "cImg2", "cPh2",
            "cCardOutput", "cOutputTag", "cDecisionBanner", "cDecisionTxt",
            "cConfTxt", "cBarDefect", "cBarOk", "castBlurVal", "castBrightVal", "castUncertVal",
            # Plastic Module (Secondary Module: dropzone, presets & defect table)
            "plasticUploadArea", "plasticFileInput",
            "pGateBadge", "pDecisionBanner", "pDecisionTxt", "pConfTxt",
            "pBarDefect", "pBarNormal", "pKpiDefects", "pKpiThreshold", "pKpiDim", "pKpiLatency",
            "btnPlasticDefect", "btnPlasticOrig",
            "pImg1", "pPh1", "pImg2", "pPh2", "pCardOutput", "pOutputTag",
            "pDefectTableCard", "pDefectTableBody"
        ]

        missing_ids = [elem_id for elem_id in required_ids if f'id="{elem_id}"' not in html]
        self.assertEqual(
            len(missing_ids), 0,
            f"DOM elements referenced in JS missing from HTML: {missing_ids}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
