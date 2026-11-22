# Technical Assumptions & Architectural Decisions

This document records key decisions, dataset findings, and assumptions made during the implementation of the **Casting Defect Detector MVP** in accordance with the project requirements.

---

## 1. Dataset Audit Findings

- **Location:** Local `./dataset/` directory.
- **Structure Discovered:**
  - `dataset/casting_512x512/casting_512x512/`:
    - `def_front`: 781 unaugmented images (512x512 px, 3-channel RGB where R=G=B).
    - `ok_front`: 519 unaugmented images (512x512 px, 3-channel RGB where R=G=B).
    - Total: 1,300 original high-resolution casting images.
  - `dataset/casting_data/casting_data/`:
    - `train/def_front`: 3,758 images (300x300 px).
    - `train/ok_front`: 2,875 images (300x300 px).
    - `test/def_front`: 453 images (300x300 px).
    - `test/ok_front`: 262 images (300x300 px).
    - Total: 7,348 pre-split augmented images.
- **Label Mapping:**
  - `ok_front` -> `OK` (Normal, defect-free casting)
  - `def_front` -> `Defective` (Casting exhibiting surface or shape defects)
- **Image Properties:**
  - All channels (R, G, B) are identical in value (grayscale stored as 24-bit JPEG).
  - Therefore, single-channel (grayscale) preprocessing is optimal and mathematically equivalent, while cutting CNN computational cost by 3x.

---

## 2. Custom Pure-NumPy CNN Architecture

- **Constraint:** Strictly NO PyTorch, TensorFlow, Keras, or OpenCV.
- **Implementation Strategy:**
  - Vectorized operations using standard NumPy.
  - Optimized forward & backward convolution utilizing `im2col` / `col2im` matrix transformations to avoid slow Python `for` loops.
  - Baseline architecture:
    1. Input: `(N, 1, 64, 64)` normalized grayscale image.
    2. Conv2D: 8 filters of shape `(3, 3)`, stride 1, padding 1 -> ReLU -> output `(N, 8, 64, 64)`.
    3. MaxPool2D: `(2, 2)`, stride 2 -> output `(N, 8, 32, 32)`.
    4. Conv2D: 16 filters of shape `(3, 3)`, stride 1, padding 1 -> ReLU -> output `(N, 16, 32, 32)`.
    5. MaxPool2D: `(2, 2)`, stride 2 -> output `(N, 16, 16, 16)`.
    6. Flatten: `(N, 16 * 16 * 16) = (N, 4096)`.
    7. Dense: `4096 -> 64` -> ReLU.
    8. Dense: `64 -> 2` (logits for `[OK, Defective]`).
    9. Numerically stable Softmax (`x - max(x)` before exponentiation).
- **Weight Serialization:** Stored as compressed NumPy arrays (`.npz`) paired with versioned metadata (`.json`).

---

## 3. Decision Policy & Uncertainty ("Perlu Review")

- **Classes:** Binary `[OK, Defective]`.
- **Threshold Policy:**
  - If model max score is >= `CONFIDENCE_THRESHOLD` (default 0.70) AND image quality check passes:
    - Status: `classified`
    - Label: `OK` or `Defective`
  - If model max score is < `CONFIDENCE_THRESHOLD` (uncertain prediction):
    - Status: `review_required` ("Perlu Review")
    - Reason: "Skor keyakinan model rendah (< 70%). Sistem menahan keputusan otomatis demi keselamatan mutu."
  - If image fails quality checks (extreme blur or lighting):
    - Status: `review_required` ("Perlu Review")
    - Reason: "Kualitas gambar di luar toleransi (terlalu gelap/terang/buram)."
  - If weights file is missing:
    - Status: `model_unavailable`
    - Reason: "Bobot model belum tersedia. Silakan jalankan training offline terlebih dahulu."

---

## 4. Image Quality Checks

- Implemented using pure Python and NumPy (no OpenCV):
  - **Dimension Check:** Minimum width & height configurable (default 100x100 px).
  - **Brightness Check:** Mean pixel luminance must be between 25 and 235 on [0, 255] scale.
  - **Blur Detection Proxy:** Variance of discrete 2D Laplacian kernel applied via 2D convolution or finite differences. Images with variance below threshold (e.g., < 15.0) receive a review recommendation.

---

## 5. Storage & Privacy

- Uploaded images for inference are processed entirely in memory via `io.BytesIO`.
- No uploaded user images are permanently saved to disk unless explicitly enabled via `ENABLE_IMAGE_PERSISTENCE = True`.
- Temporary safe filenames use random UUID4 tokens.
- SQLite database (`inspection_history.db`) stores lightweight metadata only (timestamp, inspection ID, status, score, model version, and operator notes).

---

## 6. Defect Localization (Bounding Boxes & Marker Dots)

- **Methodology:** Weakly supervised localization using analytic Pure-NumPy Grad-CAM on convolutional feature maps.
- **Why this approach?**
  - The dataset provides only binary image-level labels (`ok_front` vs `def_front`) without bounding box annotations.
  - Using Grad-CAM leverages the gradients of the `Defective` logit propagated back to the second convolutional layer (`Conv2D` with 16 filters).
  - Spatial activations are upscaled to the original image dimensions, thresholded, and clustered using 8-connectivity connected components.
  - For each detected defect cluster, the system generates:
    1. A bounding box `[x, y, width, height]` (kotak indikator).
    2. Hotspot center marker dots (titik puncak anomali).
    3. Perimeter indicator points along the defect boundary.
  - Visualization is rendered both on the frontend (interactive switcher between clean and localized views) and as a base64-annotated image.

