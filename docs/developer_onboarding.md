# Panduan Onboarding Pengembang (Developer Onboarding Guide)
Sistem Inspeksi Visual Terpadu Manufaktur (Unified Inspection Platform)

Dokumen ini disusun sebagai panduan teknis bagi tim pengembang Bearing/Casting dan tim pengembang Plastic Defect untuk melanjutkan pengembangan model, eksperimen, dan integrasi tanpa merusak kestabilan aplikasi terpadu.

---

## 1. Arsitektur Terpadu (Unified Architecture Overview)

Proyek ini menggunakan pola arsitektur decoupled adapter:
1. **Frontend Web & API Gateway (`unified_app/`)**: Mengelola antarmuka pengguna berbasis browser, pemrosesan kamera live-feed, navigasi modul (Ring, Bearing, Plastic), routing REST API (`/api/v1/...`), serta manajemen modal.
2. **Lapisan Adaptor (`unified_app/adapters/`)**: Menjembatani request API dengan modul model inspeksi independen. Masing-masing modul memiliki adaptor sendiri (`ring_adapter.py`, `bearing_adapter.py`, `plastic_adapter.py`).
3. **Pusat Riset & Pengembangan Model (`model_development/`)**: Folder kerja independen untuk masing-masing tim melatih model, mengaudit dataset, dan membangun pipeline inferensi:
   - `model_development/ring_inspection/`
   - `model_development/bearing_inspection/`
   - `model_development/plastic_inspection/` (standby)

---

## 2. Panduan Pengembang: Tim Bearing & Coran Logam

### 2.1 Struktur Folder Modul Bearing
Seluruh kode eksperimen dan inferensi modul Bearing/Casting berada di:
`Unified_Inspection/model_development/bearing_inspection/`

Berkas utama:
- `app/model_loader.py`: Mengatur pemuatan bobot model Pure-NumPy CNN (`ModelManager`).
- `app/inference_service.py`: Pipeline inferensi lengkap, perhitungan skor blur Laplacian, kecerahan, dan Grad-CAM.
- `cnn/model.py`: Arsitektur Sequential CNN Pure-NumPy (64x64 input).
- `cnn/localization.py`: Ekstraksi titik anomali dan kotak pembatas Grad-CAM tanpa library deteksi objek berat.
- `cnn/train.py`: Skrip pelatihan mandiri.
- `audit_dataset.py`: Skrip verifikasi kuantitas citra cacat dan normal.

### 2.2 Cara Mengembangkan Model Lebih Lanjut
1. **Melakukan Retraining atau Tuning**:
   Jalankan skrip pelatihan di folder bearing:
   ```powershell
   cd model_development/bearing_inspection
   python cnn/train.py
   ```
2. **Memperbarui Ambang Batas (Threshold) & Sensitivitas**:
   Pengaturan ambang deteksi dan sensitivitas QC dapat dimodifikasi pada `app/inference_service.py` di fungsi `run_casting_inspection()`.
3. **Pengujian Integrasi**:
   Setelah mengubah kode di `bearing_inspection/`, jalankan QA test suite untuk memastikan adaptor tetap kompatibel:
   ```powershell
   python tests/test_qa_suite.py
   ```

---

## 3. Panduan Pengembang: Tim Inspeksi Plastik (Plastic Defect)

Modul Plastik saat ini berada dalam status **Standby Scaffold**. Antarmuka web, tab navigasi, dan endpoint API sudah siap menerima modul tanpa perlu merombak aplikasi utama.

### 3.1 Langkah Integrasi Model Plastik
Untuk mengaktifkan modul plastik secara penuh, lakukan langkah-langkah berikut:

#### Langkah 1: Tempatkan Berkas Model & Dataset
Buat struktur berkas di folder modul plastik:
`Unified_Inspection/model_development/plastic_inspection/`
Contoh isi:
- `models/`: Tempat file bobot model (misal `plastic_cnn.pth` atau `model.onnx`).
- `inference.py`: Modul pemroses citra plastik dan prediksi cacat.
- `dataset/`: Aset dataset pengujian atau sampel validasi.

#### Langkah 2: Perbarui Adaptor Plastik
Buka berkas `unified_app/adapters/plastic_adapter.py`.
Ubah status dan lengkapi fungsi `load()` serta `inspect()`:
```python
import importlib
from .base import BaseInspectionAdapter

class PlasticAdapter(BaseInspectionAdapter):
    module_id = "plastic"
    display_name = "Plastic Defect Inspection"
    category = "Plastic Injection"

    def __init__(self):
        self.is_ready = False
        self.load_error = None
        self.model = None
        self.load()

    def load(self) -> None:
        try:
            # Impor modul inferensi plastik Anda
            plastic_infer = importlib.import_module("model_development.plastic_inspection.inference")
            self.model = plastic_infer.load_model()
            self.is_ready = True
        except Exception as e:
            self.is_ready = False
            self.load_error = str(e)

    def health(self) -> dict:
        return {
            "module_id": self.module_id,
            "display_name": self.display_name,
            "category": self.category,
            "status": "ready" if self.is_ready else "error",
            "capabilities": ["classification", "defect_localization"]
        }

    def inspect(self, image_bytes: bytes, **kwargs) -> dict:
        # Jalankan prediksi model Anda
        # Pastikan output mengembalikan: status, decision, confidence, original_image, annotated_image
        ...
```

#### Langkah 3: Lengkapi Tampilan Antarmuka (Opsional)
Jika tim plastik memiliki komponen UI khusus (misal grafik distribusi warna atau visualisasi gelembung cacat):
- Perbarui template: `unified_app/templates/modules/plastic/workspace.html`
- Perbarui skrip kontroler: `unified_app/static/js/plastic.js`
- Perbarui gaya CSS: `unified_app/static/css/plastic.css`

---

## 4. Prosedur Uji Mutu (Quality Assurance Testing)

Sebelum melakukan commit atau penggabungan fitur baru ke cabang utama (`master`):
1. Pastikan server aplikasi berjalan normal:
   ```powershell
   python run.py
   ```
2. Jalankan seluruh rangkaian tes otomatis:
   ```powershell
   python tests/test_qa_suite.py
   ```
3. Kriteria lolos uji minimal:
   - 15 pengujian dalam `test_qa_suite.py` harus menghasilkan status `OK` (0 failures, 0 errors).
   - Seluruh endpoint API merespons dengan format JSON yang sesuai kontrak.
   - Tidak ada error sintaks impor dinamis pada IDE/linter.
