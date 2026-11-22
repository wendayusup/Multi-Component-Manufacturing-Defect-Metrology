# Casting Defect Detector (Visual Inspection MVP)

A lightweight, robust, and responsive AI-based visual inspection web application for preliminary quality screening of industrial casting products.

This application is built in strict adherence to the **Master Prompt** and **PRD/SRS** specifications:
- **Upload-Only MVP:** Focuses strictly on casting product image analysis from local uploads (no camera/webcam, no video streaming, no object detection).
- **Custom Pure-NumPy CNN:** The convolutional neural network (Conv2D with `im2col`, ReLU, MaxPool2D, Dense, Softmax, Cross-Entropy Loss, backpropagation, and Adam/SGD optimizers) is implemented entirely from scratch without TensorFlow, PyTorch, Keras, or OpenCV.
- **Defect Guidance & Localization:** Automatically highlights detected defect zones using bounding boxes (pengkotakan) and hotspot indicator points (titik-titik) via weakly supervised Pure-NumPy Grad-CAM without requiring external object detection frameworks.
- **Safety & Uncertainty ("Perlu Review"):** Decisions with low model confidence (< 70%) or substandard image quality are safely abstained to prevent false approvals.
- **Human QC Verification Disclaimer:** Visible on all screens and API responses to enforce decision-support compliance.

---

## 1. System Requirements & Architecture

- **Language:** Python 3.9+ (tested on Python 3.10)
- **Web Framework:** Flask
- **Core Dependencies:** `Flask`, `numpy`, `Pillow`
- **Database:** SQLite (embedded, zero-configuration)
- **Frontend:** Responsive HTML5, CSS3, Vanilla JavaScript (no external CDN or frontend build step needed)

---

## 2. Quick Start

### 2.1 Installation
Clone the repository and install the minimal dependencies:
```bash
cd D:\VibeCode\Casting_Detection
pip install -r requirements.txt
```

### 2.2 Audit Dataset
Audit the local `./dataset/` folder to view class distributions, image counts, and verify image integrity:
```bash
python audit_dataset.py
```

### 2.3 Offline Training (Optional / Pre-trained Included)
A pre-trained model is already included in `weights/casting_cnn_v1.npz`. If you wish to retrain or experiment with different parameters:
```bash
python cnn/train.py --epochs 5 --batch-size 32 --max-train 1000 --max-test 300
```
This script computes confusion matrices, per-class Precision/Recall/F1, and exports weights with SHA-256 verification to `weights/`.

### 2.4 Offline Model Evaluation
Evaluate the active model weights on the test set:
```bash
python cnn/evaluate.py --samples 200
```

### 2.5 Run the Web Application
Start the local server:
```bash
python run_local.py
```
Open your web browser at **`http://127.0.0.1:5000`** to access the Casting Defect Detector interface.

---

## 3. Automated Testing

Run the full test suite (32 unit & integration tests) covering upload validation, image quality checks, pure-NumPy CNN math, model loading, and API contracts:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

All 32 tests pass out of the box with zero failures.

---

## 4. API Endpoints

### 4.1 `POST /api/inspect/casting`
Uploads a single casting image (`.jpg`, `.jpeg`, `.png`) for inspection.

**Request:** `multipart/form-data` with field `image`.

**Response (JSON):**
```json
{
  "inspection_id": "422961e6-b605-430c-abeb-97996c561bcf",
  "created_at": "2026-11-07T14:20:00Z",
  "module": "casting_defect",
  "component_type": "casting",
  "status": "classified",
  "predicted_label": "OK",
  "confidence_score": 0.9412,
  "scores": {
    "OK": 0.9412,
    "Defective": 0.0588
  },
  "model_version": "casting-cnn-0.1.0",
  "quality_status": "passed",
  "quality_details": {
    "dimensions": [300, 300],
    "mean_brightness": 132.4,
    "blur_score": 45.2
  },
  "processing_ms": 32.1,
  "review_reasons": [],
  "warnings": [
    "Alat visual screening eksperimental berbasis AI untuk pendukung keputusan awal. Hasil bukan jaminan sertifikasi mutu dan wajib diverifikasi oleh personel/operator QC yang berkualifikasi."
  ]
}
```

### 4.2 `GET /api/history?limit=15`
Retrieves recent inspection history logs.

### 4.3 `POST /api/inspections/<inspection_id>/review`
Allows a quality control inspector to record a manual verification outcome (`VERIFIED_OK`, `VERIFIED_DEFECTIVE`, `ESCALATED_ENGINEERING`) and notes without overwriting the original AI prediction.

### 4.4 `GET /api/model/status`
Returns the status, version, and metadata of the currently loaded CNN model.

---

## 5. Repository Structure

```text
Casting_Detection/
├── app/
│   ├── __init__.py           # Flask application factory
│   ├── routes.py             # HTTP routes & REST API endpoints
│   ├── validation.py         # File size, extension, & magic-byte validation
│   ├── image_io.py           # Safe decoding & secure filename generation
│   ├── preprocessing.py      # Shared preprocessing (grayscale, resize, norm)
│   ├── quality_check.py      # Dimension, illumination, & blur proxy checks
│   ├── inference_service.py  # Inspection orchestration & decision gates
│   ├── model_loader.py       # Weights loader & test model injector
│   ├── storage.py            # SQLite database for audit history & operator notes
│   ├── templates/
│   │   ├── base.html         # Base responsive HTML layout
│   │   └── index.html        # Main QC inspection dashboard
│   └── static/
│       ├── css/style.css     # Clean industrial QC styling
│       └── js/app.js         # Vanilla JS drag-and-drop & API communication
├── cnn/
│   ├── __init__.py
│   ├── layers.py             # Conv2D (im2col), ReLU, MaxPool2D, Flatten, Dense
│   ├── model.py              # Sequential container & baseline architecture
│   ├── losses.py             # Numerically stable Softmax & CrossEntropy
│   ├── optimizer.py          # Adam & SGD optimizers
│   ├── dataset_loader.py     # Offline dataset discovery & loader
│   ├── train.py              # Offline training CLI
│   └── evaluate.py           # Offline evaluation & confusion matrix CLI
├── dataset/                  # Local casting dataset directory
├── data/                     # SQLite database & runtime data
├── weights/                  # Trained weights (.npz) & metadata (.json)
├── tests/
│   ├── test_upload_validation.py
│   ├── test_preprocessing.py
│   ├── test_quality_check.py
│   ├── test_cnn_layers.py
│   ├── test_model_loader.py
│   └── test_inference_api.py
├── docs/
│   ├── requirements.md       # Full requirements specification
│   └── ASSUMPTIONS.md        # Technical assumptions & decisions
├── audit_dataset.py          # Dataset audit CLI tool
├── config.py                 # Central configuration
├── config.example.py         # Example configuration
├── requirements.txt          # Minimal project dependencies
├── run_local.py              # Server launcher
└── README.md                 # Project documentation
```

---

## 6. Disclaimer

> **IMPORTANT:** This application is an experimental visual screening decision-support prototype. It does not replace formal certified quality control inspection procedures or certified non-destructive testing (NDT). All predictions must be reviewed by qualified quality assurance personnel.

