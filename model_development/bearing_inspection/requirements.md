# Casting Defect Detector — MVP Requirements

**Document status:** MVP implementation requirements  
**Basis:** Master Prompt and PRD/SRS documents  
**Current scope:** Casting defect detection from uploaded image files only  
**Out of scope:** Plastic detector, maintenance detector, camera capture, real-time video, YOLO/object detection, ERP/MES integration

---

## 1. Product Overview

Casting Defect Detector is a lightweight, responsive web application for preliminary visual quality inspection of industrial casting products. A user uploads a single image from their local machine, the system verifies and validates the file, executes image quality checks, applies normalized preprocessing, executes a custom pure-NumPy CNN inference pipeline, and presents the inspection result with decision support confidence and human verification guidelines.

The tool is an experimental decision-support utility and does not replace formal certified QC procedures or qualified inspection personnel.

---

## 2. Functional Requirements Summary

| ID | Priority | Requirement | Acceptance Criteria |
|---|---|---|---|
| FR-CAST-001 | MUST | Dedicated Casting Defect Detector interface | UI clearly identifies casting inspection as the only active module. |
| FR-CAST-002 | MUST | Single-image upload per request | Supports JPG/JPEG and PNG image upload from local device. |
| FR-CAST-003 | MUST | Strict format and magic-byte validation | Validates file extension and decoded image headers via Pillow. |
| FR-CAST-004 | MUST | Server-side file integrity and size enforcement | Maximum upload size 10 MB (configurable); corrupt or empty files rejected. |
| FR-CAST-005 | MUST | Client-side preview and replace | Preview selected image before submission; remove or replace file at any time. |
| FR-CAST-006 | MUST | Consistent preprocessing | Preprocessing pipeline (decode, grayscale, resize, normalize) identical in training and inference. |
| FR-CAST-007 | MUST | Custom Pure-NumPy CNN Pipeline | No dependency on PyTorch, TensorFlow, OpenCV, or Keras for model training or inference. |
| FR-CAST-008 | MUST | Domain classification (OK vs Defective) | Binary classification output with confidence scores for each class. |
| FR-CAST-009 | MUST | Safety and uncertainty abstention ("Perlu Review") | Triggers if confidence is below threshold (< 0.70 default) or image quality fails. |
| FR-CAST-010 | MUST | Missing model weights handling | Gracefully displays "Model Belum Tersedia" without crashing or faking predictions. |
| FR-CAST-011 | MUST | Human verification disclaimer | Prominently displayed on all result screens and API responses. |
| FR-CAST-012 | MUST | No stale state on error | Request failures immediately clear prior predictions and notify the user. |
| FR-CAST-013 | SHOULD | Local inspection history and manual notes | SQLite storage recording recent inspections and operator manual reviews. |
| FR-CAST-014 | MUST | Offline dataset loader & audit CLI | Loads from `./dataset/` without modifying originals; reports class counts and corrupt files. |

---

## 3. Input & Preprocessing Specifications

- **Supported Formats:** `.jpg`, `.jpeg`, `.png`
- **Max File Size:** 10 MB (configurable via `MAX_UPLOAD_SIZE_BYTES`)
- **Default Image Preprocessing:**
  - Mode: Grayscale (1 channel)
  - Target Resolution: Configurable (default 64x64 or 128x128 for efficient pure-NumPy inference)
  - Normalization: Scale [0, 255] to [0.0, 1.0]
  - Tensor Shape: `(batch_size, 1, height, width)`
- **Quality Gates:**
  - Minimum resolution: 100x100 pixels (configurable)
  - Brightness proxy: Reject/review if average intensity < 25 (underexposed) or > 230 (overexposed)
  - Blur proxy: Laplacian variance proxy to flag overly blurred images

---

## 4. API Specification

### Endpoint: `POST /api/inspect/casting`
- **Content-Type:** `multipart/form-data`
- **Field:** `image` (file)

#### Response Schema (JSON):
```json
{
  "inspection_id": "uuid-string",
  "module": "casting_defect",
  "component_type": "casting",
  "status": "classified | review_required | model_unavailable | invalid_input | error",
  "predicted_label": "OK | Defective | null",
  "confidence_score": 0.94,
  "scores": {
    "OK": 0.94,
    "Defective": 0.06
  },
  "model_version": "casting-cnn-0.1.0",
  "quality_status": "passed | review_recommended | failed",
  "quality_details": {
    "dimensions": [300, 300],
    "mean_brightness": 128.5,
    "blur_score": 42.1
  },
  "processing_ms": 38.5,
  "review_reasons": [],
  "warnings": [
    "Alat visual screening eksperimental berbasis AI. Wajib diverifikasi oleh personel/operator QC berkualifikasi."
  ]
}
```

---

## 5. Non-Functional & Safety Constraints

- **Strict Framework Restrictions:** No TensorFlow, PyTorch, Keras, OpenCV, or pre-trained models.
- **Privacy & Storage:** Uploaded files processed in-memory (or removed immediately after processing). No sensitive image paths logged.
- **Responsive Web UI:** Fully usable on desktop monitors, laptops, tablets, and mobile smartphones.
- **Test Coverage:** Complete test suite covering valid/invalid uploads, quality gates, preprocessing, CNN layers, model loader, API contracts, and error conditions.

