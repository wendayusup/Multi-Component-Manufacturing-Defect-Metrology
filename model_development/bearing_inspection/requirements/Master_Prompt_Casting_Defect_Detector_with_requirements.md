# MASTER PROMPT — Casting Defect Detector (Upload-Only MVP)

> **Instruksi penggunaan:** Salin seluruh prompt ini ke AI coding agent (Cursor, Claude Code, ChatGPT Coding Agent, atau sejenisnya). Lampirkan dua dokumen PRD/SRS sumber jika platform mendukung attachment. Prompt ini juga memuat `requirements.md` yang menjadi acuan implementasi.

---

## 1. MASTER PROMPT

You are a senior full-stack Python engineer and machine-learning engineer helping build an academic manufacturing visual-inspection prototype.

Your task is to implement **only the Casting Defect Detector MVP**, based on the requirements in the embedded `requirements.md` and the provided PRD/SRS documents.

### 1.1 Project context

The project is an AI-based manufacturing visual-inspection web application. The broader project documents describe several modules, but the current implementation is intentionally limited to:

- **Image Detector — Casting**
- User uploads a single casting-product image.
- The application validates the image, preprocesses it, runs a custom CNN inference pipeline, and displays an inspection result.
- The first-stage classification is **OK vs Defective**, with **Perlu Review** as a safety/uncertainty outcome.

This is an **upload-only MVP**. Do not implement camera capture, webcam, real-time video, plastic-product inspection, machinery maintenance inspection, or other modules.

### 1.2 Mandatory scope restrictions

Implement only what is explicitly in scope.

**In scope**
- Casting image upload from local device.
- JPG/JPEG and PNG support.
- Image preview before analysis.
- Server-side file validation.
- Basic image quality checks where feasible and documented.
- Preprocessing consistent with model training.
- Custom CNN inference.
- Result screen with prediction, score, processing time, model version, quality status, and disclaimer.
- Clear error handling.
- Minimal project structure and tests.
- A clean responsive interface for laptop and mobile browser.

**Out of scope — do not build**
- Plastic defect detector.
- Machinery fault / maintenance detector.
- Webcam, browser camera, camera permission, frame capture, video streaming, or real-time analysis.
- YOLO or object detection.
- Bounding boxes, segmentation, defect localization, or pixel-level heatmaps unless an appropriate annotated dataset and model are later provided.
- User authentication / roles unless explicitly requested later.
- ERP, MES, CMMS, PLC, machine control, shutdown commands, or production-line integration.
- RUL (Remaining Useful Life) prediction.
- Training CNN during a user’s upload request.
- Claims of certified QC, guaranteed accuracy, or automatic quality approval.

### 1.3 Source-of-truth rules

1. Treat the uploaded PRD/SRS documents and this `requirements.md` as the source of truth.
2. If a detail is not specified, do not invent a business requirement. Use a conservative, configurable implementation and record the assumption in `docs/ASSUMPTIONS.md`.
3. If the documents conflict, prioritize the narrower current scope: **Casting Defect Detector using upload only**. Explain any conflict in `docs/ASSUMPTIONS.md`.
4. Do not silently claim a feature is complete if it is only a placeholder.
5. Do not fabricate model accuracy, dataset size, labels, or evaluation results.
6. If no casting dataset or trained model weights are provided, build the application pipeline and clearly report that inference is unavailable until real weights are trained and installed. Do not generate fake predictions or use random weights as if they were a trained model.
7. Keep the app usable in development even when weights are missing: show a clear “Model belum tersedia” / “Model unavailable” message rather than crashing.

### 1.3.1 Dataset folder requirement (MANDATORY)

**The casting dataset must be loaded from the project’s local `dataset/` folder.** Do not ask the user to upload training images through the web interface, and do not fetch a dataset from the internet unless the user explicitly requests it.

- Before coding, inspect the repository and locate the existing `dataset/` folder (case-sensitive path as present in the repository).
- Treat `dataset/` as the default source for offline training, evaluation, and dataset audit scripts. Make the path configurable with `DATASET_DIR` in the project configuration or environment variable, defaulting to `dataset/` relative to the project root.
- Do not hardcode absolute paths tied to a developer’s computer.
- Inspect the actual folder structure and labels before implementing the dataset loader. Do not assume class names, image counts, or nesting that have not been verified.
- Support a documented class-folder layout when present, for example `dataset/OK/` and `dataset/Defective/`, while adapting to the real folder names and recording the mapping in the dataset configuration. If the directory has another structure, report it and implement a loader appropriate to that structure rather than silently rearranging files.
- Recursively discover supported image files inside the configured dataset folder if the verified dataset layout requires nested directories. Ignore unsupported files and report skipped/corrupt files in the audit output.
- Dataset loading must preserve each image’s label based on the verified folder/metadata mapping. Never infer labels from filenames unless the dataset documentation explicitly defines that convention.
- Add a dataset audit script that reports discovered class labels, image counts per class, supported formats, unreadable files, and a sample of paths (relative paths only). Do not print private absolute paths unnecessarily.
- Split the dataset into train/validation/test sets without leakage; do not copy or move the original dataset files. Keep split manifests reproducible and configurable.
- The web upload feature remains for **inference only**. The `dataset/` folder is for offline training/evaluation and must not be exposed as a public static directory or downloadable web resource.
- If `dataset/` is missing or empty, do not fabricate data or silently use sample/random images. Report the missing dataset clearly, scaffold the loader/audit tooling, and continue only with parts that do not require dataset files.

### 1.4 Required technology and AI constraints

Use the following baseline unless the source documents or user explicitly change it:

- Python
- Flask
- HTML, CSS, vanilla JavaScript
- NumPy for numerical array operations
- Pillow only for image decoding/encoding if needed
- SQLite only if inspection history is implemented in this MVP
- Custom CNN implemented by the team: convolution, activation, pooling, dense layers, loss, backpropagation, optimizer, training loop, and inference logic must not be delegated to a CNN framework.

**Do not use as the main AI inference/training framework:**
- YOLO
- OpenCV
- TensorFlow
- PyTorch
- Keras
- Pretrained CNN models

Do not add these dependencies “for convenience.” Browser file upload does not require OpenCV.

The source PRD/SRS recommends offline training (local computer or notebook) and deploying only model weights plus inference code to PythonAnywhere. Follow this separation.

### 1.5 Model and label behavior

The casting pilot uses a single casting product/domain and binary labels:

- `OK`
- `Defective`

The UI/API may also return an application status:

- `classified`: model returned a valid classification.
- `review_required`: image quality failed, model score is below the validated threshold, or the model cannot safely classify the image.
- `model_unavailable`: model weights/configuration are missing or incompatible.
- `error`: invalid request or processing failure.

`Perlu Review` is **not a third training class** unless the dataset explicitly contains such a class. It is a review status produced by quality/uncertainty rules.

The displayed model score is not a guarantee of correctness and must not be called a calibrated probability unless calibration has been tested.

### 1.6 Workflow

Build the project in this order:

1. Inspect the existing repository before changing files.
2. Summarize the current structure and identify existing code that can be reused.
3. Inspect and audit the local `dataset/` folder; verify its actual structure and labels.
4. Create or update `requirements.md` from the embedded specification.
5. Propose a concise implementation plan and file structure.
5. Implement upload UI and preview.
6. Implement server-side validation and image preprocessing.
7. Implement the custom CNN inference interface and model loader.
8. Implement result rendering and clear error states.
9. Add unit and integration tests.
10. Run tests and fix issues.
11. Provide setup and run instructions.
12. Summarize implemented features, unimplemented items, assumptions, and how to train/load a real model.

Do not skip directly to a large code dump. Work incrementally and keep the application runnable after each milestone.

### 1.7 Upload and image safety

- Accept only JPG/JPEG and PNG for the MVP.
- Validate both the extension and the actual decoded image format; do not trust filename or MIME type alone.
- Enforce a configurable maximum upload size. Use **10 MB as the initial upper limit**, consistent with the PRD/SRS baseline, but make it easy to lower for hosting limits.
- Reject corrupt, empty, unsupported, or oversized files with a user-friendly message.
- Generate a safe server-side filename; never use the original filename as a filesystem path.
- Do not execute or serve uploaded files as static content.
- Avoid storing uploaded images by default unless the user explicitly enables image retention.
- If image retention is implemented, keep uploads in a private directory and document retention/cleanup.
- Do not log image contents or sensitive paths.
- Do not use a previous request’s prediction when the current request fails.

### 1.8 Image preprocessing and quality checks

The preprocessing implementation must be consistent between training and inference.

Implement a single reusable preprocessing function/configuration used by both:

- offline training/evaluation scripts; and
- web inference.

The preprocessing pipeline should be configurable and documented. It may include:
- image decoding;
- RGB or grayscale conversion based on dataset audit;
- resize/crop to the model’s configured input shape;
- normalization;
- conversion to the array format expected by the custom CNN.

Do not silently choose a final input shape if the dataset has not been audited. Use a documented baseline configuration and make it configurable.

For image quality:
- Check whether the image can be decoded.
- Check image dimensions against a configurable minimum.
- Optionally implement simple brightness and blur proxies if they can be explained and tested.
- Do not claim quality checks reliably detect all blur, lighting, or defect visibility problems.
- If a quality gate rejects the image, do not run inference and do not display a fabricated defect/normal prediction.
- Show a clear reason and suggest uploading a clearer image.

### 1.9 CNN implementation

The broader PRD/SRS describes a custom CNN baseline. Use it as a starting point, not as a guarantee that the architecture is final.

A baseline may include:
1. Input image tensor.
2. Conv2D + ReLU.
3. MaxPool2D.
4. Conv2D + ReLU.
5. MaxPool2D.
6. Flatten.
7. Dense + ReLU.
8. Dense output layer.
9. Softmax / score computation.

The team must implement and test the necessary operations manually:
- convolution forward;
- ReLU forward/backward;
- max-pooling forward/backward;
- flatten;
- dense forward/backward;
- numerically stable softmax;
- cross-entropy;
- backpropagation;
- SGD (momentum optional);
- save/load model weights.

Use NumPy for numerical operations if permitted by the project. Do not use a ready-made CNN framework.

For this current MVP, prioritize **inference integration** and a clear model interface. Training scripts can be separate from the web app. Do not train on every upload.

### 1.10 Result screen

For a successful classification, display:
- “Casting Defect Inspection” title;
- uploaded image preview;
- predicted label (`OK` or `Defective`);
- model score / confidence score;
- model version;
- processing time;
- input quality status;
- a disclaimer that this is an experimental visual screening tool and must be verified by a qualified human/operator.

For `review_required`, display:
- “Perlu Review”;
- reason (e.g. low model score or image quality issue);
- recommendation to re-upload or perform manual inspection;
- no definitive OK/Defective label if the system’s uncertainty policy says it should abstain.

For `model_unavailable`, display a clear message that a trained casting model has not been configured. Do not show a sample/random prediction as a real result.

### 1.11 API and code quality

Use a simple Flask endpoint for casting upload inference, for example:

`POST /api/inspect/casting`

The exact route may be adjusted to fit the existing project, but keep it clear and consistent.

The endpoint should:
- accept one image file;
- validate the request;
- validate/decode the image;
- run preprocessing;
- load/use the casting model;
- return a structured response;
- handle exceptions safely.

Example response shape (adapt as needed):

```json
{
  "inspection_id": "generated-id",
  "module": "casting_defect",
  "component_type": "casting",
  "status": "classified",
  "predicted_label": "defective",
  "scores": {
    "ok": 0.18,
    "defective": 0.82
  },
  "model_version": "casting-cnn-0.1.0",
  "quality_status": "passed",
  "processing_ms": 1430,
  "warnings": [
    "Prototype result; verify manually"
  ]
}
```

The values above are illustrative only. Never hardcode them as a real prediction.

Use:
- clear module separation;
- configuration rather than magic constants;
- readable names and type hints where appropriate;
- safe error handling;
- tests for valid and invalid cases;
- no secrets in source control.

### 1.12 UI/UX expectations

Create a polished but practical manufacturing QC interface:
- responsive layout for desktop and mobile;
- clear page heading and casting-only context;
- upload dropzone and browse button;
- supported format and file-size information;
- preview before submitting;
- remove/replace image option;
- loading state while inference runs;
- disabled submit button during request;
- clear success, review, model-unavailable, and error states;
- accessible labels, keyboard focus, and readable contrast;
- avoid excessive animations or decorative complexity.

Do not display navigation tabs for modules that are not implemented. If future modules are mentioned, mark them as “Coming later” only if the user specifically wants them visible.

### 1.13 Testing requirements

Write tests for at least:
- valid JPG upload;
- valid PNG upload;
- unsupported file type;
- corrupt image;
- empty upload;
- oversized upload;
- invalid dimensions;
- quality gate rejection;
- missing model weights;
- model/config mismatch;
- inference success with a test/fake model injected into the test environment;
- response structure;
- no stale prediction after a failed request.

A fake model may be used in unit tests only. Clearly distinguish test fixtures from real trained model weights.

### 1.14 Deliverables

At completion, provide:
1. Working source code for the casting upload-only MVP.
2. `requirements.md`.
3. `README.md` with environment setup, run commands, model/weights setup, and known limitations.
4. `requirements.txt` with only necessary dependencies.
5. Tests and instructions to run them.
6. `docs/ASSUMPTIONS.md` listing unresolved decisions.
7. A concise implementation summary:
   - completed;
   - not implemented;
   - dataset/model prerequisites;
   - test results;
   - local run instructions.

Start by inspecting the repository and confirming the current implementation scope. If a dataset, class mapping, model weights, or training configuration is missing, ask for it only when needed; otherwise scaffold the system safely without inventing it.

---

# 2. `requirements.md`

## Casting Defect Detector — MVP Requirements

**Document status:** MVP implementation requirements  
**Basis:** User-provided PRD/SRS documents dated 25 October 2026  
**Current scope:** Casting defect detection from uploaded image files only  
**Out of scope:** Plastic detector, maintenance detector, camera capture, real-time video

## 2.1 Product overview

Casting Defect Detector is a responsive web application for preliminary visual quality inspection of a casting product. A user uploads one image, the system validates the image, preprocesses it, runs a custom CNN model, and displays an inspection result.

The application is an experimental decision-support tool. It does not replace formal QC procedures or qualified human inspection.

## 2.2 Goals

- Allow users to upload a casting product image from a laptop or mobile browser.
- Validate file type, size, readability, and basic image suitability.
- Run inference using a custom CNN trained for the casting domain.
- Display a clear classification result and associated metadata.
- Provide a “Perlu Review” outcome when the image or model result is not suitable for a confident classification.
- Keep the implementation simple, testable, and ready for later expansion without implementing future modules now.

## 2.3 User and primary use case

**Primary user:** Operator, student, or inspector performing preliminary casting visual inspection.

**Primary flow:**
1. Open the Casting Defect Detector page.
2. Select an image file.
3. Preview the image.
4. Submit it for inspection.
5. The server validates and preprocesses the image.
6. The custom CNN performs inference if a compatible trained model is available.
7. The application displays the result or an appropriate review/error message.

## 2.3.1 Dataset directory and loading requirements (MUST)

The project must read the casting training/evaluation images from a local folder named **`dataset/`** at the repository root by default.

- Default path: `./dataset/` (relative to project root).
- Configurable override: `DATASET_DIR` environment variable or equivalent project configuration.
- Dataset loading is intended for offline training, validation, testing, and audit scripts—not for the live web upload/inference endpoint.
- Inspect the actual directory tree before deciding the loader’s class mapping. A common expected layout is:

```text
project-root/
├── dataset/
│   ├── OK/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── Defective/
│       ├── image_001.jpg
│       └── ...
```

This is an example only. Use the real folder names and nesting found in the provided dataset. Record the mapping from folder/metadata labels to model classes in a configuration file. Do not rename or move the original images automatically.

### Dataset loader and audit behavior

- Implement a reusable dataset loader for offline training/evaluation.
- Discover JPG/JPEG and PNG images from the configured dataset folder; support nested folders when needed by the actual dataset layout.
- Derive labels only from verified directory structure or dataset metadata. Do not guess labels from filenames.
- Validate image readability and report corrupted/unsupported files rather than silently treating them as valid samples.
- Provide a dataset audit command/script that reports: dataset path (relative where possible), class mapping, image count per class, total valid images, unreadable/skipped files, and detected formats.
- Keep original dataset files unchanged. Generate split manifests or references rather than duplicating/moving the source images.
- Use reproducible train/validation/test splits and prevent leakage across related images, products, batches, or acquisition sessions when such grouping information exists.
- If the folder is missing, empty, or does not match the expected label structure, show a clear actionable error and document the actual structure. Do not generate synthetic/random data or fake labels to make training appear successful.
- Never expose `dataset/` as a Flask static folder or public download path.

## 2.4 Functional requirements

| ID | Priority | Requirement | Acceptance criteria |
|---|---|---|---|
| FR-CAST-001 | MUST | The application shall provide a Casting Defect Detector page. | The page clearly identifies casting inspection as the only active module. |
| FR-CAST-002 | MUST | The user shall upload one image file per inspection request. | A single JPG/JPEG or PNG file can be selected and previewed. |
| FR-CAST-003 | MUST | The application shall validate file type. | Unsupported formats are rejected with a readable message. |
| FR-CAST-004 | MUST | The backend shall validate the actual image content, not only extension/MIME. | Renamed non-image files and corrupt images are rejected. |
| FR-CAST-005 | MUST | The application shall enforce a configurable upload size limit. | Initial maximum is 10 MB; oversized files are rejected before inference. |
| FR-CAST-006 | MUST | The UI shall show a preview and allow replacing/removing the selected image before submission. | User can review the selected file and change it. |
| FR-CAST-007 | MUST | The backend shall preprocess the image using the model’s configured preprocessing. | The same preprocessing configuration is used by training and inference. |
| FR-CAST-008 | MUST | The system shall use a casting-specific model/configuration. | Casting endpoint never accidentally loads plastic or maintenance model weights. |
| FR-CAST-009 | MUST | The system shall return `OK` or `Defective` when inference succeeds and the prediction passes the configured decision policy. | Result label is one of the two configured casting labels. |
| FR-CAST-010 | MUST | The system shall support an abstention/review state. | Low score, invalid image quality, unsupported input, or model mismatch does not force a definitive classification. |
| FR-CAST-011 | MUST | The result shall display the predicted label, score, model version, processing time, and quality status. | Required fields appear for a successful inference. |
| FR-CAST-012 | MUST | The result shall display a human-verification disclaimer. | User is told that this is preliminary AI screening, not certified QC. |
| FR-CAST-013 | MUST | Missing or incompatible model weights shall be handled gracefully. | User sees “Model belum tersedia” or equivalent; no random/fake result is displayed. |
| FR-CAST-014 | MUST | Errors shall not show a previous request’s prediction. | Each request’s result state is reset and tied to its own request. |
| FR-CAST-015 | SHOULD | The application may save inspection metadata in a minimal history. | If implemented, record includes timestamp, module, label/status, score, model version, and processing time. |
| FR-CAST-016 | SHOULD | The user may add a manual note or review outcome. | Manual review is stored separately and does not overwrite AI prediction. |
| FR-CAST-017 | COULD | The application may export inspection records. | CSV/PDF export is not required for this MVP. |

## 2.5 Out-of-scope requirements

The following must not be implemented in this MVP:

- Plastic defect inspection.
- Machinery fault or maintenance inspection.
- Webcam/browser camera access.
- Real-time frame analysis or video streaming.
- Conveyor inspection automation.
- YOLO/object detection.
- Bounding boxes, segmentation, or defect localization.
- ERP/MES/CMMS/PLC integration.
- Machine control or automatic shutdown.
- Remaining Useful Life (RUL) prediction.
- Multi-user authentication/role management unless requested separately.
- Training the CNN inside the web request.

## 2.6 Input requirements

| ID | Requirement | Baseline |
|---|---|---|
| INPUT-001 | Supported file formats | JPG/JPEG and PNG |
| INPUT-002 | Maximum upload size | 10 MB configurable; may be reduced for hosting |
| INPUT-003 | Image decoding | Must be decoded successfully by image I/O layer |
| INPUT-004 | Minimum dimensions | Configurable; use 640×480 as a provisional baseline from the PRD/SRS and verify against dataset |
| INPUT-005 | Input source | Local file upload only |
| INPUT-006 | Upload count | One image per inference request |
| INPUT-007 | File naming | Server-generated safe name if stored; never trust user filename as path |
| INPUT-008 | Storage | Do not persist image by default; if enabled, use private storage and retention policy |

If the dataset requires different image dimensions, document the change and ensure it is reflected in both training and inference.

## 2.7 Image quality and preprocessing

The system should check:
- file decoding success;
- image dimensions;
- optionally brightness/clipping;
- optionally blur proxy.

Quality thresholds must be configurable and validated against real casting images. A simple brightness/blur metric must not be represented as a perfect image-quality detector.

Preprocessing must be shared by training and inference and include, as applicable:
- RGB/grayscale conversion;
- resize/crop;
- normalization;
- tensor/array conversion;
- input shape configuration.

Do not apply inference-only transformations that were absent during training.

## 2.8 AI/model requirements

### Domain and classes
- Domain: one casting product type/domain selected by dataset audit.
- Initial classes: `OK`, `Defective`.
- The dataset’s actual class mapping must be verified before training.
- Do not invent defect subclasses or claim defect localization without annotated data.

### Model implementation
- Custom CNN implemented by the project team.
- NumPy may be used for array/numerical calculations.
- Pillow may be used for image decode/encode.
- Do not use YOLO, OpenCV, TensorFlow, PyTorch, Keras, or pretrained CNNs as the main prediction pipeline.

### Training and evaluation
- Training runs offline on a developer machine or notebook.
- The web app only loads model weights and runs inference.
- Dataset must be audited for label quality, duplicates, class balance, and usage/license.
- Split data before augmentation.
- Baseline split: train/validation/test = 70/15/15, subject to dataset suitability.
- Prevent data leakage by grouping related images by object, lot, session, or source where applicable.
- Augmentation may only be applied to the training set and must remain physically plausible.
- Report confusion matrix, precision, recall, F1 per class, macro F1, accuracy, sample counts, and threshold.
- Freeze decision threshold using validation data before final test evaluation.
- Do not fabricate or overstate model performance.

### Model metadata
Store or provide:
- `model_version`;
- `module/domain`;
- `input_shape`;
- `class_names`;
- preprocessing configuration;
- decision threshold / uncertainty policy;
- training dataset version;
- seed and training parameters;
- evaluation metrics;
- weight file checksum.

## 2.9 Result and uncertainty policy

The application may use these result statuses:

| Status | Meaning |
|---|---|
| `classified` | Model produced an accepted OK/Defective classification. |
| `review_required` | Image quality or model uncertainty requires human review. |
| `model_unavailable` | Casting weights/configuration are missing or incompatible. |
| `invalid_input` | File or image validation failed. |
| `error` | Unexpected processing failure. |

“Perlu Review” is an application status, not a third model class unless the dataset explicitly defines it.

The model’s raw score must be labeled **model score** or **confidence score**. Do not claim it is a calibrated probability unless calibration has been evaluated.

## 2.10 API requirements

Suggested endpoint:

`POST /api/inspect/casting`

Input:
- multipart/form-data;
- one image file.

Successful response should include:
- `inspection_id`;
- `module` (e.g. `casting_defect`);
- `component_type` (e.g. `casting`);
- `status`;
- `predicted_label` when classification is accepted;
- `scores`;
- `model_version`;
- `quality_status`;
- `processing_ms`;
- `warnings`.

Example schema only:

```json
{
  "inspection_id": "generated-id",
  "module": "casting_defect",
  "component_type": "casting",
  "status": "classified",
  "predicted_label": "OK",
  "scores": {
    "OK": 0.91,
    "Defective": 0.09
  },
  "model_version": "casting-cnn-0.1.0",
  "quality_status": "passed",
  "processing_ms": 1200,
  "warnings": [
    "Prototype result; verify manually"
  ]
}
```

This example is not a real model output. Do not hardcode it.

## 2.11 Nonfunctional requirements

| ID | Area | Requirement |
|---|---|---|
| NFR-001 | Responsive UI | Usable on laptop and mobile browser. |
| NFR-002 | Usability | Upload, preview, submit, and result states are clear. |
| NFR-003 | Reliability | Invalid uploads and model failures do not crash the app. |
| NFR-004 | Security | Validate file content and size; use safe file handling; do not expose private uploads. |
| NFR-005 | Maintainability | Separate routes, validation, preprocessing, model inference, and UI. |
| NFR-006 | Reproducibility | Record model version and preprocessing/model configuration. |
| NFR-007 | Performance | Measure upload, preprocessing, inference, and total response time. Do not claim real-time performance without evidence. |
| NFR-008 | Safety | Clearly state that AI output is preliminary and requires human QC verification. |
| NFR-009 | Storage | Avoid unnecessary image retention and keep hosting usage within limits. |
| NFR-010 | Accessibility | Accessible form labels, keyboard focus, readable contrast, and clear error text. |

## 2.12 UI requirements

The MVP page should contain:
- application title;
- casting-only module label;
- short instruction;
- accepted file formats and max size;
- upload area and browse button;
- image preview;
- remove/replace file action;
- analyze button;
- loading indicator;
- result panel;
- error/review/model-unavailable states;
- disclaimer.

Do not show inactive module tabs unless explicitly requested.

## 2.13 Testing and acceptance criteria

Minimum test cases:

| Test ID | Scenario | Expected result |
|---|---|---|
| TEST-001 | Upload valid JPG | File accepted and previewed. |
| TEST-002 | Upload valid PNG | File accepted and previewed. |
| TEST-003 | Upload unsupported format | Rejected with clear message. |
| TEST-004 | Upload corrupt image | Rejected; inference not called. |
| TEST-005 | Upload empty file | Rejected. |
| TEST-006 | Upload file above limit | Rejected. |
| TEST-007 | Image below minimum dimensions | Rejected or review warning according to configured policy. |
| TEST-008 | Quality gate fails | Inference not called; show reason. |
| TEST-009 | Model weights missing | Show model unavailable; no fake prediction. |
| TEST-010 | Model configuration mismatch | Fail safely and log a non-sensitive error. |
| TEST-011 | Valid input with test model | Response schema and result UI work. |
| TEST-012 | Inference exception | Show error; do not show stale result. |
| TEST-013 | Mobile browser | Upload and result flow remain usable. |
| TEST-014 | Human-verification disclaimer | Visible on result page. |

A fake model may be injected for automated tests only. Never use it as the production model.

## 2.14 Suggested repository structure

```text
casting-defect-detector/
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── validation.py
│   ├── image_io.py
│   ├── preprocessing.py
│   ├── quality_check.py
│   ├── inference_service.py
│   ├── model_loader.py
│   ├── storage.py                 # optional for history
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   └── result.html            # optional; result may be in index
│   └── static/
│       ├── css/
│       └── js/
├── cnn/
│   ├── layers.py
│   ├── losses.py
│   ├── optimizer.py
│   ├── model.py
│   ├── train.py                   # offline only
│   └── evaluate.py                # offline only
├── dataset/                       # casting dataset source folder (local/private)
│   ├── OK/                        # example only; use verified actual labels
│   └── Defective/                  # example only; use verified actual labels
├── datasets/
│   └── README.md                  # optional audit/split manifests; do not duplicate source images
├── weights/                        # trained weights, versioned
├── tests/
│   ├── test_upload_validation.py
│   ├── test_preprocessing.py
│   ├── test_quality_check.py
│   ├── test_inference_api.py
│   └── test_model_loader.py
├── docs/
│   ├── requirements.md
│   └── ASSUMPTIONS.md
├── requirements.txt
├── config.example.py
├── run_local.py
└── README.md
```

Adjust this structure if the existing repository already has a sensible layout. Do not rewrite unrelated files.

## 2.15 Definition of Done

The MVP is considered complete when:

- [ ] Only Casting Defect Detector is implemented.
- [ ] The user can upload JPG/PNG and preview the selected image.
- [ ] File type, actual content, size, and decode validation work.
- [ ] Preprocessing is shared/configured for training and inference.
- [ ] Offline dataset loader reads from the configured local `dataset/` folder.
- [ ] Dataset audit reports classes, counts, and unreadable files without inventing labels.
- [ ] Casting model loading is separate from web routes.
- [ ] Missing weights are handled without fake predictions.
- [ ] Successful inference displays label, score, model version, quality status, and processing time.
- [ ] Review/uncertain and error states are implemented.
- [ ] Disclaimer is visible.
- [ ] Automated tests pass.
- [ ] README explains setup, running, testing, and model installation.
- [ ] Known limitations and unresolved assumptions are documented.
- [ ] No out-of-scope modules or camera features have been added.

## 2.16 Open decisions to document (do not guess silently)

- Exact casting product type and dataset source.
- Actual `dataset/` directory structure and whether labels are folder-based or metadata-based.
- Dataset class names and label mapping.
- Dataset license/permission.
- Actual number of images per class.
- RGB vs grayscale.
- Final input image size.
- Quality gate thresholds.
- CNN architecture and hyperparameters after baseline testing.
- Decision threshold for `review_required`.
- Whether inspection history is needed for the MVP.
- Whether uploaded images should be retained.
- Deployment resource constraints and hosting plan.

Record these in `docs/ASSUMPTIONS.md` and ask the project owner only when the decision blocks implementation.

---

## 3. Catatan penting sebelum coding

- **Jangan membuat prediksi palsu** jika dataset atau model weights belum tersedia.
- **Jangan membuat fitur kamera** karena MVP ini hanya upload file.
- **Jangan memasukkan modul plastik atau maintenance** hanya karena tercantum di dokumen proyek besar.
- **Jangan mengklaim CNN sudah akurat** sebelum evaluasi dataset dilakukan.
- **Jangan menyamakan image classification dengan object detection**. MVP hanya mengklasifikasikan satu gambar casting sebagai OK/Defective.
