# Automated Visual Inspection Platform
## Manufacturing Component Defect Classification & Metrology Station

A production-grade, self-contained automated visual inspection system integrating deep learning and computer vision metrology for real-time industrial defect detection.

---

## Executive Summary

This platform consolidates three independent inspection modules into a unified enterprise interface:
1. **Bearing Ring Inspection (`ring`)**: High-resolution 512x512 surface metrology powered by a **Pure 7-Stage Deep CNN** (PyTorch) with Dual-Gate Safety Interlock (CNN + Sobel Texture Gradient) and 262,144 analytical cells disparity matrix (**95.83% accuracy, 100% defect precision / zero false rejects**).
2. **Casting Metal Inspection (`bearing`)**: Built from scratch using a **Pure-NumPy Sequential CNN** (64x64 Grayscale, zero external deep learning framework) with weakly-supervised Grad-CAM defect localization and 128x128 Surface Anomaly Matrix (**92.80% accuracy**).
3. **Plastic Defect Inspection (`plastic`)**: Multi-scale **Feature Pyramid Network (FPN) CNN** (PyTorch, 768x768 scale-preserved canvas padding) with center-point regression, bounding box geometry heads, and Non-Maximum Suppression (**IoU 0.45 NMS**).

---

## Project Architecture & Directory Structure

Everything is self-contained within this repository. Development from scratch, model training, and the web station coexist cleanly:

```text
Unified_Inspection/
├── run.py                                     # Main execution entrypoint for Flask web app
├── README.md                                  # Single comprehensive project documentation
│
├── docs/                                      # OFFICIAL SPECIFICATIONS & CONTRACTS
│   ├── integration_requirements.md            # System integration, adapter contract & SLA
│   └── ui_ux_requirements.md                 # Enterprise UI/UX design system & tokens
│
├── model_development/                         # MODEL TRAINING & RESEARCH FROM SCRATCH
│   ├── ring_inspection/                      # [RING TEAM] Model dev, training, & datasets
│   │   ├── dataset_raw/                      # Raw primary photos taken directly from workshop
│   │   ├── dataset_split/                    # Split data (train, val, test)
│   │   ├── saved_models/pure_cnn_best.pth    # Trained PyTorch model weights (512x512)
│   │   ├── step01_data_preparation/          # Augmentation & CLAHE preprocessing
│   │   ├── step02_model_architecture/        # Pure CNN 7-Stage architecture definition
│   │   ├── step03_training/train.py          # Training loop script
│   │   ├── step04_evaluation/evaluate.py     # Evaluation & confusion matrix generator
│   │   ├── step05_inference/                 # DefectLocalizer & 512x512 disparity matrix
│   │   └── run_web_app.py                    # Standalone Ring testing web app
│   │
│   ├── bearing_inspection/                   # [BEARING/CASTING TEAM] Model dev & tests
│   │   ├── casting_512x512/                  # Metal casting image dataset
│   │   ├── weights/casting_cnn_v1.npz        # Pure-NumPy model weights (2 MB)
│   │   ├── weights/casting_cnn_v1_meta.json  # Model metadata & evaluation metrics
│   │   ├── cnn/                              # Manual Conv2D, MaxPool, ReLU, Grad-CAM (NumPy)
│   │   ├── tests/                            # 7 automated unit test suites
│   │   └── run_local.py                      # Standalone Casting testing web app
│   │
│   └── plastic_inspection/                   # [PLASTIC TEAM] Model dev & weights
│       ├── models/detector.py                # FPN Single-Class CNN architecture
│       ├── models/plastic_defect_detector.pt # Trained PyTorch checkpoint (877 KB)
│       ├── models/environment.json           # Model metadata and input parameters
│       └── samples/                          # Defect and normal sample datasets
│
└── unified_app/                               # UNIFIED WEB PLATFORM & INFERENCE ENGINE
    ├── __init__.py                           # Flask application factory (`create_app`)
    ├── config.py                             # Central application configuration & dynamic paths
    │
    ├── adapters/                             # Isolated Model Adapter Layer
    │   ├── base.py                           # BaseInspectionAdapter abstract interface
    │   ├── ring_adapter.py                   # Connects model_development/ring_inspection
    │   ├── bearing_adapter.py                # Connects model_development/bearing_inspection
    │   └── plastic_adapter.py                # Connects model_development/plastic_inspection
    │
    ├── routes/                               # API & Page Routing
    │   ├── pages.py                          # Single Page Application root route (`/`)
    │   └── api.py                            # RESTful inspection endpoints (/api/v1/inspect/*)
    │
    ├── static/                               # Modular Static Assets
    │   ├── css/
    │   │   ├── base.css                      # Global SaaS theme & design tokens
    │   │   ├── ring.css                      # [RING TEAM] Ring workspace styling
    │   │   ├── bearing.css                   # [BEARING TEAM] Bearing workspace styling
    │   │   └── plastic.css                   # [PLASTIC TEAM] Plastic workspace styling
    │   └── js/
    │       ├── app.js                        # Shared controller (nav switcher, camera, modals)
    │       ├── ring.js                       # [RING TEAM] Ring UI logic & real-time loop
    │       ├── bearing.js                    # [BEARING TEAM] Bearing UI logic & Grad-CAM toggle
    │       └── plastic.js                    # [PLASTIC TEAM] Plastic UI controller
    │
    └── templates/                            # Modular Jinja2 Templates
        ├── index.html                        # Master shell layout
        └── modules/
            ├── ring/workspace.html           # [RING TEAM] HTML layout for Ring tab
            ├── bearing/workspace.html        # [BEARING TEAM] HTML layout for Bearing tab
            └── plastic/workspace.html        # [PLASTIC TEAM] HTML layout for Plastic tab
```

---

## Formal Specification Documents

The platform strictly adheres to two comprehensive engineering specifications:
1. **[Integration Requirements Specification](docs/integration_requirements.md)**: Architectural decoupling rules, adapter interface contracts (`BaseInspectionAdapter`), API endpoints, dataset paths, and latency/accuracy SLAs.
2. **[UI/UX Requirements Specification](docs/ui_ux_requirements.md)**: Enterprise design tokens, one-page desktop layout, eye-level verdict banner placement, dynamic 6-step metrology slider, dual side-by-side viewports, and zero-flicker real-time camera streaming.

---

## Team Ownership & Git Collaboration

Each team member works in their designated workspace without conflicting with others:

| Module | Member | Model Dev & Training | HTML Template | CSS Stylesheet | JS Controller |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bearing Ring** | Member 1 | `model_development/ring_inspection/` | `templates/modules/ring/workspace.html` | `static/css/ring.css` | `static/js/ring.js` |
| **Casting Metal** | Member 2 | `model_development/bearing_inspection/` | `templates/modules/bearing/workspace.html` | `static/css/bearing.css` | `static/js/bearing.js` |
| **Plastic Defect**| Member 3 | `model_development/plastic_inspection/`| `templates/modules/plastic/workspace.html`| `static/css/plastic.css`| `static/js/plastic.js` |

---

## Quick Start Guide

### 1. Prerequisites
Ensure Python 3.10+ is installed along with required packages:
```bash
pip install flask torch torchvision opencv-python pillow numpy
```

### 2. Running the Unified Web Station
From the workspace root:
```bash
python Unified_Inspection/run.py
```
Access the application in your browser at:
- **URL**: `http://localhost:5000`

---

## Standalone Model Development & Training

### Bearing Ring (`ring_inspection`):
```bash
cd Unified_Inspection/model_development/ring_inspection

# Train Pure CNN from scratch:
python step03_training/train.py

# Evaluate test accuracy & confusion matrix:
python step04_evaluation/evaluate.py

# Run standalone web app:
python run_web_app.py
```

### Casting Metal (`bearing_inspection`):
```bash
cd Unified_Inspection/model_development/bearing_inspection

# Run full automated unit test suite:
python -m unittest discover tests

# Run standalone web app:
python run_local.py
```

---

## REST API Specifications

### Discovery & Health
- `GET /api/v1/health` - Platform status and readiness of all models.
- `GET /api/v1/modules` - List of available inspection modules and their capabilities.

### Inference Endpoints
- `POST /api/v1/inspect/ring` - Run Bearing Ring inspection (Accepts multipart image file or base64 JSON, with `grid_dim` parameter).
- `POST /api/v1/inspect/bearing` - Run Casting Metal inspection (Accepts multipart image file or base64 JSON, with `sensitivity` parameter).
- `POST /api/v1/inspect/plastic` - Run Plastic Defect inspection (Accepts multipart image file or base64 JSON).
- `POST /api/v1/inspect/all` - Sequential cross-module inspection on a single component image.

### Sample Presets & Evaluation Figures
- `GET /api/sample/ring/<defect|normal>` - Preset sample images for Ring testing.
- `GET /api/sample/bearing/<defect|normal>` - Preset sample images for Casting testing.
- `GET /api/sample/plastic/<defect|normal>` - Preset sample images for Plastic testing.
- `GET /api/analytics/ring/confusion_matrix` - Serves academic confusion matrix plot.
- `GET /api/analytics/ring/training_history` - Serves academic training loss curve plot.
