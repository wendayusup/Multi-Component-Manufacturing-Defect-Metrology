"""
Image Quality Assessment Layer.
Evaluates input image suitability: dimension limits, illumination levels,
and sharpness/blur proxy. Flags poor quality inputs for human review.
"""
from typing import Dict, Tuple
import numpy as np
from PIL import Image

import config


def compute_blur_proxy(gray_array: np.ndarray) -> float:
    """
    Computes a sharpness/blur metric using discrete 2D Laplacian variance.
    Pure NumPy implementation without OpenCV.
    """
    if gray_array.shape[0] < 3 or gray_array.shape[1] < 3:
        return 0.0

    # Discrete 5-point Laplacian stencil
    center = gray_array[1:-1, 1:-1]
    top = gray_array[:-2, 1:-1]
    bottom = gray_array[2:, 1:-1]
    left = gray_array[1:-1, :-2]
    right = gray_array[1:-1, 2:]

    laplacian = top + bottom + left + right - 4.0 * center
    variance = float(np.var(laplacian))
    return round(variance, 2)


def check_image_quality(image: Image.Image) -> Dict:
    """
    Inspects image quality metrics and returns quality status and warnings:
    - Dimensions check (width, height)
    - Luminance / exposure check
    - Sharpness / blur proxy check
    """
    width, height = image.size
    warnings = []
    is_hard_fail = False

    # 1. Dimension Check
    if width < config.QUALITY_MIN_WIDTH or height < config.QUALITY_MIN_HEIGHT:
        warnings.append(
            f"Dimensi gambar terlalu kecil ({width}x{height}px). "
            f"Batas minimum yang disarankan adalah {config.QUALITY_MIN_WIDTH}x{config.QUALITY_MIN_HEIGHT}px."
        )
        is_hard_fail = True

    # 2. Brightness Check
    gray_img = image.convert("L")
    gray_arr = np.array(gray_img, dtype=np.float32)
    mean_brightness = float(np.mean(gray_arr))

    if mean_brightness < config.QUALITY_MIN_BRIGHTNESS:
        warnings.append(
            f"Gambar terlalu gelap (kecerahan rata-rata {mean_brightness:.1f}/255). "
            f"Harap gunakan pencahayaan yang cukup."
        )
    elif mean_brightness > config.QUALITY_MAX_BRIGHTNESS:
        warnings.append(
            f"Gambar terlalu terang / silau (kecerahan rata-rata {mean_brightness:.1f}/255). "
            f"Hindari overexposure pada permukaan casting."
        )

    # 3. Blur Check
    blur_score = compute_blur_proxy(gray_arr)
    if blur_score < config.QUALITY_BLUR_THRESHOLD:
        warnings.append(
            f"Tingkat ketajaman gambar rendah (skor blur: {blur_score:.1f}, ambang: {config.QUALITY_BLUR_THRESHOLD}). "
            f"Gambar terindikasi buram/kabur."
        )

    if is_hard_fail:
        status = "failed"
    elif len(warnings) > 0:
        status = "review_recommended"
    else:
        status = "passed"

    return {
        "passed": status == "passed",
        "status": status,
        "dimensions": [width, height],
        "mean_brightness": round(mean_brightness, 2),
        "blur_score": blur_score,
        "warnings": warnings
    }

