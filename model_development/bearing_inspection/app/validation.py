"""
Upload Request and File Validation Layer.
Enforces file format, magic numbers, file size limits, and non-empty content.
"""
from typing import Tuple
from werkzeug.datastructures import FileStorage

import config


# Magic bytes signatures for supported image types
MAGIC_SIGNATURES = {
    "jpeg": [b"\xff\xd8\xff"],
    "png": [b"\x89PNG\r\n\x1a\n"]
}


def validate_file_extension(filename: str) -> bool:
    """Checks if the file extension is allowed."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS


def validate_magic_bytes(header: bytes) -> bool:
    """Validates that initial bytes match JPEG or PNG magic signatures."""
    if len(header) < 4:
        return False

    for signatures in MAGIC_SIGNATURES.values():
        for sig in signatures:
            if header.startswith(sig[:len(header)]):
                return True
    return False


def validate_upload(file_storage: FileStorage) -> Tuple[bytes, str]:
    """
    Validates uploaded FileStorage object:
    - Verifies file exists and has a filename
    - Verifies extension
    - Checks file size within limit
    - Checks magic bytes
    Returns raw bytes and sanitized original filename.
    """
    if file_storage is None:
        raise ValueError("Tidak ada file yang diunggah. Silakan pilih file gambar.")

    filename = file_storage.filename or ""
    if not filename.strip():
        raise ValueError("Nama file kosong. Silakan pilih file gambar yang valid.")

    if not validate_file_extension(filename):
        allowed = ", ".join([f".{e}" for e in config.ALLOWED_EXTENSIONS])
        raise ValueError(f"Ekstensi file tidak didukung. Format yang diizinkan: {allowed}")

    # Read content
    file_bytes = file_storage.read()

    if len(file_bytes) == 0:
        raise ValueError("File yang diunggah kosong (0 bytes).")

    if len(file_bytes) > config.MAX_CONTENT_LENGTH:
        max_mb = config.MAX_CONTENT_LENGTH / (1024 * 1024)
        raise ValueError(f"Ukuran file melebihi batas maksimum {max_mb:.0f} MB.")

    # Validate magic bytes
    if not validate_magic_bytes(file_bytes[:16]):
        raise ValueError("Isi file tidak sesuai dengan format JPEG atau PNG yang valid.")

    return file_bytes, filename

