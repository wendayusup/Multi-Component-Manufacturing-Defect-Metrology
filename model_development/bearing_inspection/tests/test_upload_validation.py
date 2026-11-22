"""
Unit Tests for Upload and File Validation Layer.
Verifies format checks, magic byte headers, empty files, and size restrictions.
"""
import unittest
import io
from werkzeug.datastructures import FileStorage

from app.validation import (
    validate_file_extension,
    validate_magic_bytes,
    validate_upload
)
import config


class TestUploadValidation(unittest.TestCase):

    def test_valid_extensions(self):
        """TEST-001 & TEST-002: Accepts jpg, jpeg, png."""
        self.assertTrue(validate_file_extension("casting_01.jpg"))
        self.assertTrue(validate_file_extension("casting_02.jpeg"))
        self.assertTrue(validate_file_extension("casting_03.png"))
        self.assertTrue(validate_file_extension("CASTING_04.PNG"))

    def test_unsupported_extensions(self):
        """TEST-003: Rejects disallowed file types."""
        self.assertFalse(validate_file_extension("sample.txt"))
        self.assertFalse(validate_file_extension("payload.exe"))
        self.assertFalse(validate_file_extension("image.gif"))
        self.assertFalse(validate_file_extension("report.pdf"))
        self.assertFalse(validate_file_extension("no_extension"))

    def test_magic_bytes_detection(self):
        """Verifies binary header detection."""
        jpeg_header = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        fake_header = b"NOT AN IMAGE FILE"

        self.assertTrue(validate_magic_bytes(jpeg_header))
        self.assertTrue(validate_magic_bytes(png_header))
        self.assertFalse(validate_magic_bytes(fake_header))

    def test_empty_file_rejected(self):
        """TEST-005: Rejects 0-byte file."""
        empty_storage = FileStorage(stream=io.BytesIO(b""), filename="empty.jpg")
        with self.assertRaises(ValueError) as ctx:
            validate_upload(empty_storage)
        self.assertIn("kosong", str(ctx.exception).lower())

    def test_renamed_non_image_rejected(self):
        """TEST-004: Rejects text file disguised as jpg."""
        fake_storage = FileStorage(stream=io.BytesIO(b"Hello world this is text"), filename="fake.jpg")
        with self.assertRaises(ValueError) as ctx:
            validate_upload(fake_storage)
        self.assertIn("tidak sesuai", str(ctx.exception).lower())

    def test_oversized_file_rejected(self):
        """TEST-006: Rejects file exceeding maximum configured size."""
        oversized_bytes = b"\xff\xd8\xff" + b"0" * (config.MAX_CONTENT_LENGTH + 1024)
        oversized_storage = FileStorage(stream=io.BytesIO(oversized_bytes), filename="huge.jpg")
        with self.assertRaises(ValueError) as ctx:
            validate_upload(oversized_storage)
        self.assertIn("melebihi batas", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()

