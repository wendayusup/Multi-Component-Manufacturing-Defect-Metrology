# Product Requirements Document (PRD) & Software Requirements Specification (SRS)
## Automated Visual Inspection Platform (`Unified_Inspection`)

**Versi Dokumen:** 1.0  
**Tanggal:** 25 November 2026  
**Mata Kuliah:** Tugas Besar Artificial Intelligence (AIT) — Politeknik Manufaktur Bandung  
**Tim Pengembang:**
- **Wenda Maulana Yusup** (224443046 / 3AEC2) — Modul *Bearing Ring Inspection* (`ring`)
- **Muhammad Putra Nugraha** (224443014 / 3AEC1) — Modul *Casting Metal / Bearing Inspection* (`bearing`)
- **Ardika Fakhrizain Hawari** (224443026 / 3AEC2) — Modul *Plastic Defect Inspection* (`plastic`)

---

# BAGIAN I: PRODUCT REQUIREMENTS DOCUMENT (PRD)

## 1. Latar Belakang & Tujuan Produk
Pada proses manufaktur industri, pemeriksaan kualitas visual (*Quality Control* / QC) secara manual rentan terhadap kelelahan operator, subjektivitas penilaian, dan keterbatasan kecepatan inspeksi. Sebelumnya, masing-masing model deteksi cacat (Bearing Ring, Casting Metal, dan Plastic Injection) dikembangkan secara terpisah sehingga sulit dioperasikan dalam satu alur kerja yang seragam.

**Tujuan Produk (`Unified_Inspection`):**
1. Mengintegrasikan tiga model inspeksi visual berbasis AI ke dalam **satu platform web terpadu** (*one-stop metrology station*).
2. Menyediakan hasil keputusan QC secara *real-time* (`PASS (NORMAL)`, `REJECT (DEFECT)`, atau `PERLU REVIEW`) lengkap dengan tingkat keyakinan (*confidence*) dan visualisasi analitik.
3. Menjaga isolasi antar-modul melalui arsitektur *decoupled adapter* sehingga pengembangan model tiap anggota tim tidak saling mengganggu stabilitas aplikasi utama.

---

## 2. Target Pengguna (User Persona)
| Persona | Peran | Kebutuhan Utama |
| :--- | :--- | :--- |
| **Operator QC Lantai Produksi** | Pengguna harian di stasiun inspeksi | Antarmuka satu layar tanpa *scroll* berlebih, keputusan `PASS`/`REJECT` yang jelas di level mata, dukungan kamera *real-time* dan unggah gambar cepat. |
| **Engineer / Peneliti AI** | Pengembang & evaluator model | Akses metrik kuantitatif (latensi, probabilitas, gradien Sobel, *anomaly matrix*), grafik evaluasi (Confusion Matrix & Training History), serta API terstandar untuk pengujian otomatis. |

---

## 3. Ruang Lingkup & Fitur Utama Produk

### 3.1 Fitur Platform Terpadu
- **Navigasi Multi-Modul**: Perpindahan cepat antar tab **Bearing Ring**, **Casting / Bearing**, dan **Plastic Defect** pada satu halaman (*Single-Page Workspace*).
- **Multi-Mode Akuisisi Citra**:
  - **File Upload**: Unggah gambar komponen (*drag-and-drop* atau file picker).
  - **Preset Sample**: Tombol uji cepat sampel `Defect` dan `Normal` untuk verifikasi instan.
  - **Live Camera Streaming**: Pengambilan gambar langsung dari kamera USB/webcam dengan mode *Snapshot* maupun *Auto Real-Time Loop* (interval `600 ms`, `800 ms`, `1200 ms`) serta fitur *Flip Camera*.
- **Dual Side-by-Side Viewport**: Menampilkan citra input asli di sisi kiri dan hasil analisis/visualisasi di sisi kanan, dilengkapi *Zoom Modal*.
- **Academic Benchmarks Modal**: Menampilkan *Confusion Matrix*, kurva *Training History*, dan ringkasan arsitektur evaluasi.

### 3.2 Fitur Spesifik per Modul
1. **Bearing Ring Inspection (`ring`)**:
   - Klasifikasi permukaan *bearing ring* (512×512 CLAHE Grayscale) dengan **Pure 7-Stage Deep CNN** + **Dual-Gate Safety Interlock** (CNN + Sobel Texture Gradient).
   - **Dynamic Resolution Slider** untuk *Disparity Metrology Matrix* (`128×128` hingga `4096×4096`, default `512×512` / 262.144 sel).
   - Pilihan filter visual: *Defect Bounding Boxes*, *Disparity Metrology Matrix*, dan filter industri (*CLAHE*, *Black-Hat*, *Sobel*).
2. **Casting Metal / Bearing Inspection (`bearing`)**:
   - Klasifikasi coran logam (64×64 Grayscale) menggunakan **Pure-NumPy Sequential CNN** (tanpa framework DL eksternal).
   - *Quality Gate* otomatis (pemeriksaan dimensi minimum, kecerahan, dan ketajaman/blur Laplacian).
   - Visualisasi citra hasil dengan badge status bersih (`DEFECTIVE`) serta *128×128 Surface Anomaly Matrix* (Grad-CAM *jet heatmap*) dan pengaturan sensitivitas QC.
3. **Plastic Defect Inspection (`plastic`)**:
   - Deteksi cacat injeksi plastik (768×768 *scale-preserved canvas padding*) menggunakan **Single-Class FPN CNN** dengan *center-point regression*, *bounding box geometry*, dan *Non-Maximum Suppression (NMS)*.

---

## 4. Alur Pengguna (User Flow)
1. Pengguna membuka aplikasi web di `http://localhost:5000`.
2. Sistem memuat status kesiapan seluruh model melalui `/api/v1/health`.
3. Pengguna memilih modul inspeksi (`Bearing Ring`, `Casting / Bearing`, atau `Plastic Defect`).
4. Pengguna memilih metode input: **Upload Gambar**, **Sample Preset**, atau **Live Camera**.
5. Sistem mengirim citra ke endpoint `/api/v1/inspect/<module_id>`, menjalankan preprocessing + inferensi AI, lalu menampilkan:
   - **Verdict Banner** (`PASS` / `REJECT` / `PERLU REVIEW`) + confidence.
   - **Distribusi Probabilitas & KPI Metrik** (latensi, gradien, jumlah marker, resolusi).
   - **Dual Viewport** (citra input vs citra hasil/heatmap).

---

## 5. Metrik Keberhasilan (Success Metrics)
- **Akurasi Model**:
  - Bearing Ring: **95.83%** akurasi uji (presisi cacat **100%**).
  - Casting / Bearing: **92.80%** akurasi uji.
- **Keandalan Sistem**: Seluruh 16 kasus uji pada `tests/test_qa_suite.py` dan 34 unit test pada `model_development/bearing_inspection/tests` berstatus `OK`, serta kegagalan satu modul tidak menjatuhkan modul lain (*fault isolation*).

---
---

# BAGIAN II: SOFTWARE REQUIREMENTS SPECIFICATION (SRS)

## 1. Pendahuluan & Arsitektur Sistem

### 1.1 Pola Arsitektur (*Decoupled Model-Adapter Pattern*)
Aplikasi menggunakan arsitektur berlapis berbasis Python Flask:
1. **Presentation Layer (`unified_app/templates/`, `unified_app/static/`)**: HTML5 (Jinja2), CSS3 modular (`base.css`, `ring.css`, `bearing.css`, `plastic.css`), dan Vanilla JS (`app.js`, `ring.js`, `bearing.js`, `plastic.js`).
2. **Routing & API Gateway (`unified_app/routes/`)**: Menangani halaman web (`pages.py`) dan endpoint REST JSON (`api.py`).
3. **Service & Adapter Layer (`unified_app/services/`, `unified_app/adapters/`)**: `ModelRegistry` mendaftarkan `RingAdapter`, `BearingAdapter`, dan `PlasticAdapter` yang seluruhnya mewarisi kontrak `BaseInspectionAdapter`.
4. **Model Development Layer (`model_development/`)**: Direktori independen untuk dataset, training, evaluasi, dan kode inferensi masing-masing modul (`ring_inspection/`, `bearing_inspection/`, `plastic_inspection/`).

### 1.2 Lingkungan Sistem (*System Environment*)
- **Bahasa Pemrograman**: Python 3.10+
- **Web Framework**: Flask
- **Library AI & Computer Vision**: PyTorch, Torchvision, NumPy, OpenCV (`cv2`), Pillow (`PIL`)
- **Target Browser**: Browser modern berbasis Chromium / standar WebRTC (`getUserMedia` untuk kamera) pada resolusi desktop 1080p (responsif pada layar ≤ 1080px).

---

## 2. Kebutuhan Fungsional (Functional Requirements)

| ID | Kebutuhan Fungsional | Deskripsi Teknis |
| :--- | :--- | :--- |
| **FR-01** | **Pendaftaran & Health Check Modul** | Sistem harus memuat seluruh adapter saat startup melalui `ModelRegistry` dan menyediakan status kesiapan tiap modul pada `GET /api/v1/health` dan `GET /api/v1/modules`. |
| **FR-02** | **Inspeksi Bearing Ring** | `RingAdapter` memproses citra 512×512 CLAHE melalui Pure 7-Stage CNN (Gate 1) dan Sobel/Black-Hat (Gate 2, ambang `sobel_mean >= 26.0`), menghasilkan keputusan, `views`, serta matriks disparitas sesuai parameter `grid_dim` (`128`–`4096`). |
| **FR-03** | **Inspeksi Casting / Bearing** | `BearingAdapter` menjalankan *quality check* (dimensi, kecerahan, blur Laplacian), inferensi Pure-NumPy CNN 64×64, `Surface Anomaly Matrix` 128×128, serta merender gambar hasil dengan badge status `DEFECTIVE` (tanpa kotak/titik merah overlay) sesuai parameter `sensitivity`. |
| **FR-04** | **Inspeksi Plastic Defect** | `PlasticAdapter` memproses citra dengan *scale-preserved padding* 768×768 melalui model FPN CNN (`plastic_defect_detector.pt`) dan NMS untuk menghasilkan prediksi dan visualisasi cacat. |
| **FR-05** | **Inspeksi Lintas Modul (`inspect/all`)** | Endpoint `POST /api/v1/inspect/all` harus mampu menjalankan inspeksi berurutan pada modul aktif untuk satu citra input yang sama. |
| **FR-06** | **Input Multi-Format & Preset** | Seluruh endpoint `/api/v1/inspect/*` harus menerima unggahan `multipart/form-data` (`file`) maupun JSON `base64`, serta menyediakan endpoint sampel `GET /api/sample/<module>/<defect|normal>`. |
| **FR-07** | **Analitik & Benchmark Akademik** | Sistem menyediakan endpoint `GET /api/analytics/ring/<analytic_type>` untuk menyajikan plot *Confusion Matrix*, *Training History*, dan laporan benchmark. |
| **FR-08** | **Penanganan Input Rusak & Fault Isolation** | Sistem harus mengembalikan HTTP `400` untuk payload kosong atau *base64*/biner korup, serta mengembalikan `status: "error"` atau `"model_unavailable"` jika satu modul bermasalah tanpa menghentikan server. |

---

## 3. Spesifikasi Antarmuka & Kontrak REST API

### 3.1 Kontrak Adapter (`BaseInspectionAdapter`)
Setiap adapter wajib mengimplementasikan:
- `health() -> Dict[str, Any]`: Mengembalikan `module_id`, `display_name`, `category`, `status`, dan `capabilities`.
- `inspect(image_bytes: bytes, **kwargs) -> Dict[str, Any]`: Mengembalikan dictionary terstandar berisi `status`, `decision`, `is_defect`, `predicted_label`, `confidence`, `probabilities`, `processing_ms`, `original_image`, dan `annotated_image`.

### 3.2 Daftar Endpoint REST API
| Method | Endpoint | Fungsi |
| :--- | :--- | :--- |
| `GET` | `/` | Halaman utama aplikasi (*Single-Page Workspace*). |
| `GET` | `/inspection/ring` | Halaman workspace langsung pada tab Ring. |
| `GET` | `/inspection/bearing` | Halaman workspace langsung pada tab Bearing. |
| `GET` | `/api/v1/health` | Status kesehatan server dan kesiapan semua model. |
| `GET` | `/api/v1/modules` | Daftar modul inspeksi beserta kapabilitasnya. |
| `POST` | `/api/v1/inspect/ring` | Menjalankan inspeksi Bearing Ring (parameter opsional: `grid_dim`, `sensitivity`). |
| `POST` | `/api/v1/inspect/bearing` | Menjalankan inspeksi Casting/Bearing (parameter opsional: `sensitivity`). |
| `POST` | `/api/v1/inspect/plastic` | Menjalankan inspeksi Plastic Defect. |
| `POST` | `/api/v1/inspect/all` | Menjalankan inspeksi berurutan pada seluruh modul. |
| `GET` | `/api/sample/<module>/<defect\|normal>` | Mengambil gambar sampel preset untuk pengujian. |
| `GET` | `/api/analytics/ring/<analytic_type>` | Mengambil gambar/laporan evaluasi akademik modul Ring. |

---

## 4. Kebutuhan Non-Fungsional (Non-Functional Requirements)

| Kategori | Kode | Spesifikasi |
| :--- | :--- | :--- |
| **Performa (SLA)** | **NFR-01** | Latensi inferensi target: Bearing Ring `< 60 ms` (GPU) / `< 180 ms` (CPU); Casting/Bearing `< 35 ms` (CPU). |
| **Keandalan** | **NFR-02** | Isolasi error antar-modul (*Runtime Fault Isolation*); server tetap berjalan normal meskipun salah satu file bobot model gagal dimuat. |
| **Usabilitas (UI/UX)** | **NFR-03** | Layout desktop 1080p satu halaman (`height: 100vh`), penempatan *Verdict Banner* setinggi mata, angka tabular (`tabular-nums`) untuk mencegah pergeseran layout saat *streaming*, dan tampilan hasil tajam tanpa blur buatan. |
| **Maintainability** | **NFR-04** | Pemisahan tegas antara kode tim Ring, Bearing, dan Plastic di `model_development/`, `adapters/`, `templates/modules/`, `static/css/`, dan `static/js/`. |
| **Keamanan Input** | **NFR-05** | Validasi format citra dan dekoding *base64* yang aman sebelum diproses oleh model inferensi. |

---

## 5. Rencana Verifikasi & Pengujian (Verification Plan)
1. **Pengujian Integrasi & API (`tests/test_qa_suite.py`)**:
   - Memverifikasi 16 skenario pengujian: `/api/v1/health`, `/api/v1/modules`, endpoint sampel, inferensi *multipart* & *base64* untuk Ring/Bearing/Plastic, `/api/v1/inspect/all`, penanganan payload kosong/korup (HTTP 400), endpoint analitik, serta integritas ID elemen DOM pada HTML.
2. **Pengujian Unit Modul Bearing (`model_development/bearing_inspection/tests/`)**:
   - Memverifikasi 34 unit test mencakup layer CNN Pure-NumPy, preprocessing, *model loader*, *localization*, dan API lokal.
3. **Pengujian Manual Browser**:
   - Menjalankan `python run.py`, mengakses `http://localhost:5000`, serta menguji upload gambar, sampel preset, dan kamera *real-time* pada ketiga tab modul.
