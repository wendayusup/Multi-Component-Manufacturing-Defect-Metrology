"""
Image Input/Output and Safety Layer.
Handles safe decoding of uploaded images, format validation, and safe filename generation.
Prevents path traversal, memory attacks, and unsupported formats.
"""
import io
import uuid
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image

import config


def generate_safe_filename(original_filename: str, prefix: str = "insp_") -> str:
    """
    Generates a secure, randomized server-side filename with a validated extension.
    Never uses untrusted user paths or original filenames on disk.
    """
    ext = Path(original_filename).suffix.lower()
    if ext not in {f".{e}" for e in config.ALLOWED_EXTENSIONS}:
        ext = ".jpg"
    unique_id = uuid.uuid4().hex
    return f"{prefix}{unique_id}{ext}"


def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    """
    Safely decodes raw byte stream into a PIL Image.
    Verifies that the byte stream represents an actual, readable image.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Gambar kosong (0 bytes).")

    try:
        # Wrap in BytesIO
        stream = io.BytesIO(image_bytes)
        img = Image.open(stream)
        # Verify image integrity
        img.verify()

        # Reopen after verify (verify closes/exhausts stream in PIL)
        stream.seek(0)
        img = Image.open(stream)
        # Load pixels into memory so stream can be released
        img.load()

        # Check format
        format_lower = (img.format or "").lower()
        if format_lower not in {"jpeg", "jpg", "png"}:
            raise ValueError(f"Format gambar '{format_lower}' tidak didukung. Harap unggah JPG atau PNG.")

        return img
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"File gambar rusak atau tidak valid: {str(e)}")

