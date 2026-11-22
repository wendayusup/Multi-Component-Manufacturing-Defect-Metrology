# UI/UX Requirements Specification
## Project: Multi-Component Automated Visual Inspection Station

**Version:** 2.0 (Enterprise Industrial Metrology Standard)  
**Target Environment:** 1080p Desktop Displays & Touch/Industrial HMIs  
**Language:** Professional Technical English  
**Design Philosophy:** Clean, high-legibility enterprise SaaS interface tailored for factory floor quality control operators and metrology research defense.

---

## 1. Visual Design Tokens & Palette

### 1.1 Color Tokens
All styles MUST utilize CSS variables defined in `base.css`:

| Token | Hex Value | Semantic Purpose |
| :--- | :--- | :--- |
| `--color-canvas` | `#F4F6F9` | Neutral workstation background |
| `--color-surface` | `#FFFFFF` | Primary card background |
| `--color-surface-subtle` | `#F8FAFC` | Inner card background & table fills |
| `--color-surface-muted` | `#EFF3F8` | Inactive tabs, sliders, & metric tracks |
| `--color-text-primary` | `#0F172A` | High-contrast headings and primary labels |
| `--color-text-secondary`| `#334155` | Body copy and descriptions |
| `--color-text-muted` | `#64748B` | Sub-labels, captions, and metadata |
| `--color-border` | `#E2E8F0` | Structural card borders and dividers |
| `--primary` | `#2563EB` | Active primary blue, sliders, and highlights |
| `--color-pass` | `#059669` | Emerald green for `PASS` verdicts and normal status |
| `--color-pass-bg` | `#ECFDF5` | Light green tint for pass cards and banners |
| `--color-reject` | `#DC2626` | Vibrant red for `REJECT (DEFECT)` verdicts |
| `--color-reject-bg` | `#FEF2F2` | Light red tint for defect cards and banners |
| `--color-warn` | `#D97706` | Amber yellow for `REVIEW REQUIRED` and standby |

### 1.2 Typography
- **Font Family**: Google Font `Poppins` (Weights: 400, 500, 600, 700, 800) with fallback `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`.
- **Formatting**: Tabular numbers (`font-variant-numeric: tabular-nums`) for latencies, cell counts, and percentages to prevent layout jittering during real-time streaming.

---

## 2. Page Architecture & Layout Hierarchy

### 2.1 Strict One-Page Desktop Layout
- `html, body` MUST set `height: 100vh; overflow: hidden;` to maintain an all-in-one control console where operators can monitor feeds without vertical page displacement.
- Responsive collapse: On displays `<= 1080px`, layout collapses gracefully into a scrollable single column.

### 2.2 Top Navbar (`.app-navbar`)
1. **Left Branding**:
   - Title: `Automated Visual Inspection`
   - Subtitle: `Manufacturing Defect Metrology Station`
   - Zero institutional logos (neutral enterprise presentation).
2. **Center Module Switcher**:
   - 3 Segmented Tabs: `Bearing Ring`, `Casting / Bearing`, `Plastic Defect`.
   - Ring tab displays active green pulsing dot (`.nav-tab-dot-online`).
   - Module tag pills: `Grad-CAM` on Casting, `Standby` on Plastic.
3. **Right Action**:
   - Model Benchmarks button (`#btnBenchmarksNav`) opens the Academic Evaluation (S2) modal with confusion matrix and training curves.

---

## 3. Left Control Column Specification (`.control-column`)

The left column (fixed width `380px`, `overflow-y: auto` with thin 6px custom scrollbar) contains 4 structured cards:

### Card 1: Input & Image Acquisition
- **Segmented Switcher**: Toggle between `File Upload` and `Live Camera`.
- **Upload Mode**:
  - Drag-and-drop zone with SVG upload icon.
  - Quick sample presets: `Defect Sample` and `Normal Sample` for 1-click verification.
- **Camera Mode**:
  - Live `<video>` stream box (1:1 aspect ratio).
  - Clean video display: **NO oval focus overlays or obstructive vignette shadows**.
  - Camera device dropdown selector (supporting USB webcams & virtual capture devices).
  - Primary action buttons: `Start Real-Time` (toggle), `Snapshot`, and `Flip Camera Feed`.
  - Scan rate selector: `600 ms (Fast)`, `800 ms (Balanced)`, `1200 ms (Stable)`.

### Card 2: QC Final Verdict Banner (Eye-Level Placement)
- Positioned immediately beneath the input controls for zero operator eye-travel.
- Prominent verdict banner (`.decision-banner-box`):
  - Dynamic status icon (Checkmark for PASS, Alert Hexagon for REJECT).
  - High-contrast verdict label: `PASS (NORMAL)` or `REJECT (DEFECT)`.
  - Sub-label detailing confidence percentage and triggered safety gate (`Gate 1 (Pure CNN)` vs `Gate 2 (Metrology Interlock)`).

### Card 3: Confidence Bars & Quantitative Metrology KPIs
- **Probability Distribution**:
  - Horizontal progress bar for `Defect Probability` (Red gradient) and `Normal Probability` (Green gradient).
- **4-Cell KPI Grid**:
  1. *Defect Clusters*: Number of detected anomaly regions.
  2. *Disparity Grid*: Current active matrix resolution (e.g. `512x512`).
  3. *Sobel Gradient*: Surface texture gradient score with threshold comparison (`Th: 26.0`).
  4. *Latency*: Processing latency in milliseconds.

### Card 4: QC Compliance & Industrial Disclaimer
- Compact disclaimer card explaining the Dual-Gate metrology safety policy and operator verification requirements.

---

## 4. Right Main Viewport Column Specification (`.metrology-main-col`)

### 4.1 Header & Filter Controls
- Module title and dynamic latency pill badge.
- **Filter Pills**: One-click switching between views:
  - *Ring*: `Defect Bounding Boxes` | `Disparity Metrology Matrix` | `CLAHE Illumination`.
  - *Bearing*: `Grad-CAM Bounding Boxes` | `128x128 Surface Anomaly Matrix`.
- **Dynamic Resolution Slider (Ring Module)**:
  - Range input with 6 discrete resolution steps: `128x128`, `256x256`, `512x512`, `1024x1024`, `2048x2048`, `4096x4096`.
  - Active cell counter badge (e.g. `512x512 active (262,144 cells)`).
  - Adjusting the slider automatically re-computes the metrology grid and updates the viewport.

### 4.2 Dual Side-by-Side Viewport (`.dual-viewport-grid`)
- Two equal-width image viewport cards:
  1. **Frame 1 (Input Feed)**: Displays raw native input image with `INPUT FEED` badge and `Zoom Original` action.
  2. **Frame 2 (QC Output)**: Displays the selected analytical overlay with `QC OUTPUT` badge, `Zoom HD` action, and verdict glow:
     - Emerald border glow on `PASS`.
     - Vibrant red border glow on `DEFECT`.
- **No-Blur Requirement**: Output images MUST preserve native sharpness without artificial background blurring. Bounding boxes are rendered with crisp contrast directly over genuine defect coordinates.

### 4.3 Technical Description Card
- Bottom card presenting method background, academic references (e.g. CLAHE, Serra Mathematical Morphology, Sobel Topography), and operational explanation.

---

## 5. Modal Dialogs

### 5.1 Lightbox Zoom Modal (`#zoomModal`)
- Fullscreen high-resolution inspection viewer with dark translucent backdrop (`rgba(10, 15, 30, 0.88)`).
- Dismissible via background click or ESC / Close button.

### 5.2 Academic Evaluation (S2) Modal (`#s2Modal`)
- Displays confusion matrix plots, training history curves, and academic evaluation metrics (test accuracy, precision, recall, F1-score) to support scientific evaluation.

---

## 6. Interaction & State Isolation Rules

1. **Clean Workspace Switching**: Switching between tabs MUST immediately terminate any active camera streams, cancel recursive timer loops (`clearTimeout`), and reset processing flags to prevent cross-module interference or memory leaks.
2. **Non-Blocking Inference**: All network requests to `/api/v1/inspect/*` are asynchronous (`fetch` API). UI displays transient "Processing..." placeholders while awaiting inference.
