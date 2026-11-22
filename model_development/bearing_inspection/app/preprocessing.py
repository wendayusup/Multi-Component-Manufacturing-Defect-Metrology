"""
Image Preprocessing Pipeline.
Guarantees identical preprocessing logic across offline training and live web inference.
Converts input image to Grayscale, resizes to model input shape, normalizes,
and packages into tensor format: (1, channels, height, width).
"""
from typing import Tuple
import numpy as np
from PIL import Image

import config
from app.image_io import decode_image_bytes


def preprocess_pil_image(
    image: Image.Image,
    target_size: Tuple[int, int] = config.IMAGE_TARGET_SIZE,
    channels: int = config.IMAGE_CHANNELS,
    scale: float = config.NORMALIZATION_SCALE
) -> np.ndarray:
    """
    Transforms a PIL Image into a normalized tensor ready for the Custom CNN.
    Returns ndarray of shape (1, channels, target_height, target_width), float32.
    """
    if channels == 1:
        img_converted = image.convert("L")
    else:
        img_converted = image.convert("RGB")

    # Resize using bilinear interpolation
    img_resized = img_converted.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to float32 NumPy array and scale to [0.0, 1.0]
    arr = np.array(img_resized, dtype=np.float32) / scale

    if channels == 1:
        # arr shape: (H, W) -> expand to (1, 1, H, W)
        tensor = np.expand_dims(arr, axis=(0, 1))
    else:
        # arr shape: (H, W, C) -> transpose to (1, C, H, W)
        arr = np.transpose(arr, (2, 0, 1))
        tensor = np.expand_dims(arr, axis=0)

    return tensor


def preprocess_image_bytes(image_bytes: bytes) -> Tuple[np.ndarray, Image.Image]:
    """
    End-to-end preprocessing pipeline for byte payloads:
    Decodes image and produces preprocessed tensor alongside decoded PIL Image for metadata/preview.
    """
    image = decode_image_bytes(image_bytes)
    tensor = preprocess_pil_image(image)
    return tensor, image

