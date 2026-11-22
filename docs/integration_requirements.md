# System Integration Requirements Specification
## Project: Multi-Component Automated Visual Inspection Station

**Version:** 2.0 (Modular Enterprise Architecture)  
**Target Environment:** Python 3.10+, Flask 3.x, Modern Chromium / Web Standards  
**Scope:** Integration architecture uniting independent model development suites (`model_development/`) with the unified web station (`unified_app/`).

---

## 1. Architectural Principles

### 1.1 Decoupled Model-Adapter Pattern
1. **Zero Monolithic Coupling**: Model research, network definition, training routines, and raw dataset structures MUST reside strictly in their respective subdirectories within `model_development/`.
2. **Adapter Boundary Requirement**: The web runtime (`unified_app`) NEVER calls raw model code directly. All interactions MUST pass through a standardized adapter extending `unified_app.adapters.base.BaseInspectionAdapter`.
3. **Runtime Fault Isolation**: If an individual model crashes (e.g., CUDA OOM, corrupted weights, missing file), the adapter MUST catch the exception and return a structured JSON response with `status: "error"` or `status: "model_unavailable"`. The web server and remaining inspection modules MUST continue serving traffic without downtime.

---

## 2. Directory & Namespace Layout

The codebase enforces strict namespace isolation:

```text
Unified_Inspection/
├── run.py                                     # Entry point
├── docs/
│   ├── integration_requirements.md            # This document
│   └── ui_ux_requirements.md                 # UI/UX design specification
├── model_development/                         # Standalone ML training suites
│   ├── ring_inspection/                      # PyTorch Pure CNN 512x512
│   ├── bearing_inspection/                   # Pure-NumPy Sequential CNN 64x64
│   └── plastic_inspection/                   # Plastic Defect Scaffold
└── unified_app/                               # Unified runtime engine
    ├── adapters/                             # Adapters (Ring, Bearing, Plastic)
    ├── routes/                               # REST API & Page controllers
    ├── static/                               # Modular CSS & JS
    └── templates/                            # Modular Jinja2 layouts
```

---

## 3. Module Specifications & Contracts

### 3.1 Module 1: Bearing Ring Surface Defect (`ring`)
- **Backend Directory**: `model_development/ring_inspection/`
- **Model Architecture**: Pure 7-Stage Deep Convolutional Neural Network (PyTorch, without transfer learning).
- **Weights File**: `saved_models/pure_cnn_best.pth` (~2.3 MB).
- **Input Resolution**: `512x512 Grayscale (1 channel)`.
- **Image Preprocessing**: CLAHE (`clipLimit=2.0`, `tileGridSize=(8, 8)`), normalize `mean=[0.5], std=[0.5]`.
- **Decision Engine (Dual-Gate Safety Interlock)**:
  - *Gate 1 (Deep CNN)*: Softmax classification score between `defect_ring` and `normal_ring`.
  - *Gate 2 (Metrology Interlock)*: Gaussian blur (7×7) + Sobel spatial gradient magnitude. If `sobel_mean >= 26.0`, abnormal roughness triggers a defect flag.
  - *Decision Policy*: `REJECT (DEFECT)` if Gate 1 OR Gate 2 confirms defect; `PASS (NORMAL)` only when both confirm clearance.
- **Metrology Output**:
  - Dynamic disparity matrix scaling (`128x128`, `256x256`, `512x512`, up to `4096x4096` cells).
  - Connected component anomaly bounding boxes extracted directly from raw disparity values (zero artificial background blurring).
- **Performance SLA**: Test accuracy **95.83%**, Defect Precision **100.0%** (zero false reject rate), Latency **< 60 ms** (GPU) / **< 180 ms** (CPU).

### 3.2 Module 2: Metal Casting Defect (`bearing`)
- **Backend Directory**: `model_development/bearing_inspection/`
- **Model Architecture**: Pure-NumPy Sequential CNN (Conv2D -> ReLU -> MaxPool2D -> Conv2D -> ReLU -> MaxPool2D -> Flatten -> Dense -> ReLU -> Dense). Implemented entirely in NumPy without PyTorch or TensorFlow.
- **Weights File**: `weights/casting_cnn_v1.npz` (~2.0 MB) accompanied by `weights/casting_cnn_v1_meta.json`.
- **Input Resolution**: `64x64 Grayscale (1 channel)`.
- **Quality Gating**:
  - Image dimension check (`width >= 100`, `height >= 100`).
  - Brightness bounds (`25.0 <= mean_brightness <= 235.0`).
  - Laplacian variance blur check (`blur_score >= 15.0`).
- **Defect Localization**: Weakly-supervised Grad-CAM computed via backward gradient pass on Conv Layer 2 (`32x32` CAM upscaled to native resolution).
- **Metrology Output**: `128x128 Surface Anomaly Matrix` combining Grad-CAM activation and gradient disparity.
- **Performance SLA**: Test accuracy **92.80%**, Defect Recall **95.20%**, Latency **< 35 ms** on standard CPU.

### 3.3 Module 3: Plastic Injection Defect (`plastic`)
- **Backend Directory**: `model_development/plastic_inspection/`
- **Status**: Standby scaffold. Adapter returns `status: "standby"` until model weights are delivered.

---

## 4. Adapter Interface Definition

Every adapter MUST inherit from `BaseInspectionAdapter` and implement:

```python
class BaseInspectionAdapter(ABC):
    module_id: str
    display_name: str
    category: str

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Returns readiness, model version, capabilities, and error state."""
        pass

    @abstractmethod
    def inspect(self, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """
        Executes end-to-end inspection.
        Must return standard dictionary:
        {
            "status": "success" | "error" | "model_unavailable",
            "decision": "PASS (NORMAL)" | "REJECT (DEFECT)" | "REVIEW REQUIRED",
            "is_defect": bool,
            "confidence": float (0.0 - 100.0),
            "processing_ms": float,
            "original_image": "data:image/jpeg;base64,...",
            "annotated_image": "data:image/jpeg;base64,...",
            ...
        }
        """
        pass
```

---

## 5. REST API Contract & Route Specifications

### 5.1 Platform Discovery
- **`GET /api/v1/health`**
  - Response: `{ "status": "healthy", "all_models_ready": bool, "modules": { ... } }`
- **`GET /api/v1/modules`**
  - Response: Metadata dictionary of all active modules (`ring`, `bearing`, `plastic`).

### 5.2 Inspection Processing
- **`POST /api/v1/inspect/<module_id>`**
  - **Content-Type**: `multipart/form-data` (file upload) OR `application/json` (`{"image_base64": "data:image/jpeg;base64,..."}`).
  - **Optional Query / Form Params**:
    - `grid_dim`: int (128, 256, 512, 1024, 2048, 4096) for Ring.
    - `sensitivity`: string ("standard", "high", "ultra") for Bearing.
  - **Success Response (200 OK)**:
    ```json
    {
      "module_id": "ring",
      "status": "success",
      "decision": "REJECT (DEFECT)",
      "is_defect": true,
      "confidence": 99.95,
      "probabilities": { "defect_ring": 99.95, "normal_ring": 0.05 },
      "processing_ms": 48.2,
      "num_defect_markers": 2,
      "gate_info": "Gate 1 (Pure CNN)",
      "original_image": "data:image/jpeg;base64,...",
      "annotated_image": "data:image/jpeg;base64,...",
      "metrics": { ... }
    }
    ```

### 5.3 Quick Sample Presets & Evaluation Figures
- **`GET /api/sample/<module_id>/<sample_type>`**
  - `sample_type`: `defect` | `normal` (or `defective` | `ok`).
  - Returns binary JPEG image stream directly for instant demo and automated sanity testing.
- **`GET /api/analytics/ring/<analytic_type>`**
  - `analytic_type`: `confusion_matrix` | `training_history` | `s2_report`.
  - Serves PNG figures and academic metrics for thesis/paper defense.
