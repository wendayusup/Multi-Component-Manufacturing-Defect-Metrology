"""
Integration and API Contract Tests for Casting Defect Detector.
Tests HTTP endpoints, upload handling, schema compliance, uncertainty abstention,
and human-verification disclaimer visibility.
"""
import unittest
import io
from pathlib import Path
from PIL import Image
import numpy as np

from app import create_app
from app.model_loader import get_model_manager
from cnn.model import build_casting_cnn
import config


def create_test_image(format="JPEG", size=(300, 300), color=(128, 128, 128)) -> io.BytesIO:
    """Helper to generate a valid in-memory test image with synthetic texture."""
    arr = np.random.randint(60, 190, size=(size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf


class TestInferenceAPI(unittest.TestCase):

    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_get_index_page(self):
        """TEST-013 & TEST-014: Verifies main page returns 200 and includes viewport & disclaimer."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        # Mobile viewport meta tag
        self.assertIn('name="viewport"', html)
        # Module branding
        self.assertIn("Casting Defect Detector", html)
        # Human verification disclaimer
        self.assertIn("pendukung keputusan", html.lower())
        self.assertIn("verifikasi", html.lower())

    def test_inspect_valid_jpg(self):
        """TEST-001 & TEST-011: Uploads valid JPG and validates response contract."""
        buf = create_test_image(format="JPEG", size=(300, 300))
        data = {
            "image": (buf, "sample_casting.jpg")
        }
        res = self.client.post("/api/inspect/casting", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)

        json_data = res.get_json()
        self.assertIn("inspection_id", json_data)
        self.assertEqual(json_data["module"], "casting_defect")
        self.assertIn(json_data["status"], ["classified", "review_required", "model_unavailable"])
        self.assertIn("scores", json_data)
        self.assertIn("quality_status", json_data)
        self.assertIn("processing_ms", json_data)
        self.assertIn("warnings", json_data)

        # TEST-014: Disclaimer is always present in warnings
        self.assertTrue(any("verifikasi" in w.lower() for w in json_data["warnings"]))

    def test_inspect_valid_png(self):
        """TEST-002: Uploads valid PNG and receives 200."""
        buf = create_test_image(format="PNG", size=(300, 300))
        data = {
            "image": (buf, "sample_casting.png")
        }
        res = self.client.post("/api/inspect/casting", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertIn(json_data["status"], ["classified", "review_required", "model_unavailable"])

    def test_inspect_unsupported_format(self):
        """TEST-003: Uploads unsupported text file and receives 400."""
        buf = io.BytesIO(b"Plain text payload")
        data = {
            "image": (buf, "document.txt")
        }
        res = self.client.post("/api/inspect/casting", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertEqual(json_data["status"], "invalid_input")
        self.assertIn("tidak didukung", json_data["error_message"].lower())

    def test_inspect_corrupt_image(self):
        """TEST-004: Uploads corrupt image bytes and receives 400 invalid_input."""
        buf = io.BytesIO(b"\xff\xd8\xff\xe0" + b"CORRUPTED BYTES")
        data = {
            "image": (buf, "corrupt.jpg")
        }
        res = self.client.post("/api/inspect/casting", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertEqual(json_data["status"], "invalid_input")

    def test_inspect_empty_file(self):
        """TEST-005: Uploads 0-byte file and receives 400."""
        buf = io.BytesIO(b"")
        data = {
            "image": (buf, "empty.png")
        }
        res = self.client.post("/api/inspect/casting", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertEqual(json_data["status"], "invalid_input")

    def test_inspect_tiny_dimensions_rejected(self):
        """TEST-007: Image below minimum resolution receives invalid_input."""
        buf = create_test_image(format="JPEG", size=(50, 50))
        data = {
            "image": (buf, "tiny.jpg")
        }
        res = self.client.post("/api/inspect/casting", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertEqual(json_data["status"], "invalid_input")
        self.assertEqual(json_data["quality_status"], "failed")

    def test_no_stale_prediction_after_failure(self):
        """TEST-012: Ensure subsequent failed upload does not retain stale prediction."""
        # 1. Successful upload
        buf1 = create_test_image(format="JPEG", size=(300, 300))
        res1 = self.client.post("/api/inspect/casting", data={"image": (buf1, "ok.jpg")}, content_type="multipart/form-data")
        self.assertEqual(res1.status_code, 200)

        # 2. Failed upload
        buf2 = io.BytesIO(b"INVALID")
        res2 = self.client.post("/api/inspect/casting", data={"image": (buf2, "bad.txt")}, content_type="multipart/form-data")
        self.assertEqual(res2.status_code, 400)
        json2 = res2.get_json()
        self.assertIsNone(json2.get("predicted_label"))
        self.assertIsNone(json2.get("confidence_score"))

    def test_history_and_operator_review_flow(self):
        """Tests FR-CAST-015 and FR-CAST-016: history listing and manual review update."""
        buf = create_test_image(format="JPEG", size=(300, 300))
        res = self.client.post("/api/inspect/casting", data={"image": (buf, "test.jpg")}, content_type="multipart/form-data")
        json_data = res.get_json()
        insp_id = json_data.get("inspection_id")

        # Save operator manual review
        review_res = self.client.post(f"/api/inspections/{insp_id}/review", json={
            "operator_review": "VERIFIED_OK",
            "operator_note": "Visual check confirmed no cracks."
        })
        self.assertEqual(review_res.status_code, 200)

        # Fetch history
        hist_res = self.client.get("/api/history")
        self.assertEqual(hist_res.status_code, 200)
        hist_data = hist_res.get_json()
        self.assertGreater(hist_data["count"], 0)

        # Check that operator review is recorded
        found = any(item["inspection_id"] == insp_id and item["operator_review"] == "VERIFIED_OK" for item in hist_data["inspections"])
        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
