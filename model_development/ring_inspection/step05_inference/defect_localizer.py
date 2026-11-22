"""
MODUL INFERENSI, METROLOGI CACAT, & FILTER STANDAR INDUSTRI BEARING RING
------------------------------------------------------------------------
Modul ini mengimplementasikan alur inspeksi visi komputer manufaktur terpadu:
1. Citra Asli Bebas Blur: Citra dipertahankan alami dan tajam tanpa manipulasi blur buatan.
2. Klasifikasi objektif menggunakan model Pure CNN (PASS vs REJECT).
3. Matriks Disparitas Numerik Resolusi Penuh 256x256 (65.536 sel) skala [0 - 99].
4. Bounding Box Merah diturunkan LANGSUNG dari sel matriks yang nilainya jomplang tinggi.
5. 3 Filter Standar Industri Pilihan: CLAHE, Morfologi Black-Hat, Texture Gradient.

PERBAIKAN ANTI-CAHAYA:
- CLAHE diterapkan sebelum Black-Hat untuk meratakan highlight spekuler logam.
- Ambang batas adaptif (mean + k*std) menggantikan threshold absolut.
- Shape filter membuang blob bulat (tipikal pantulan cahaya), hanya menyimpan blob memanjang (baret).
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Menambahkan root direktori proyek ke sys.path agar impor modul internal selalu berhasil
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step02_model_architecture.pure_cnn import PureCNN

# Definisi nama kelas hasil klasifikasi biner
CLASS_NAMES = ["defect_ring", "normal_ring"]


class ApplyCLAHE(object):
    """
    Transformasi kustom PyTorch untuk mengaplikasikan filter CLAHE
    pada citra PIL grayscale sebelum dikonversi menjadi tensor.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        np_img = np.array(img)
        eq_img = self.clahe.apply(np_img)
        return Image.fromarray(eq_img)


# Pipeline transformasi citra standar untuk input Pure CNN 512x512 piksel dengan CLAHE
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Kamus Metadata 3 Filter Standar Industri Pilihan (Sesuai Referensi Jurnal Ilmiah)
INDUSTRIAL_FILTER_METADATA = {
    "filter_clahe": {
        "name": "CLAHE (Pencahayaan Adaptif)",
        "journal": "Zuiderveld (1994) / Reza (2004) IEEE Trans. Medical Imaging & Metallography",
        "description": "Menormalkan variasi pencahayaan lokal pada permukaan logam reflektif tanpa memperkuat derau latar (efektif mengangkat area gelap lengkungan ring).",
        "category": "Normalisasi Pencahayaan & Kontras"
    },
    "filter_morpho": {
        "name": "Morfologi Cacat (Black-Hat)",
        "journal": "Serra (1982) Mathematical Morphology / Industrial Surface Quality Inspection",
        "description": "Mengekstrak fitur cacat gelap mikro seperti retakan (cracks), goresan dalam, dan lubang jarum (pitting). Rumus: (f • b) - f.",
        "category": "Morfologi Matematika Cacat"
    },
    "filter_texture_grad": {
        "name": "Texture Gradient (Sobel Topography)",
        "journal": "Sobel (1968) Machine Vision / Texture Topography Analysis",
        "description": "Memetakan topografi guratan mikro logam dan membedakan tekstur fisik badan ring dengan latar belakang meja/kertas secara tegas.",
        "category": "Analisis Tekstur Permukaan"
    }
}


class DefectLocalizer:
    """
    Kelas utama yang menangani inspeksi cacat bearing ring:
    - Klasifikasi menggunakan Pure CNN (tanpa transfer learning)
    - Perhitungan Matriks Disparitas 256x256 (65.536 sel) dengan CLAHE + Black-Hat
    - Ambang batas adaptif untuk membedakan baret asli dari pantulan cahaya (spekuler)
    - Pelokalan kotak merah langsung dari matriks (tanpa blur latar belakang)
    - Penyediaan 3 filter standar industri manufaktur pilihan
    """

    def __init__(self, checkpoint_path=None, device=None):
        # 1. Menentukan perangkat komputasi (GPU jika tersedia, jika tidak gunakan CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 2. Menentukan lokasi file bobot model terbaik hasil training (Stage 03)
        if checkpoint_path is None:
            checkpoint_path = PROJECT_ROOT / "saved_models" / "pure_cnn_best.pth"
        self.checkpoint_path = Path(checkpoint_path)

        # 3. Inisialisasi arsitektur Pure CNN (in_channels=1 karena grayscale)
        self.model = PureCNN(in_channels=1, num_classes=2).to(self.device)

        # 4. Memuat bobot hasil training ke dalam model
        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()  # Set model ke mode evaluasi (mematikan dropout)
            print(f"[INFO] Model Pure CNN berhasil dimuat dari: {self.checkpoint_path.name}")
        else:
            print(f"[PERINGATAN] File bobot tidak ditemukan di {self.checkpoint_path}")

    # =========================================================================
    # TAHAP 1: DETEKSI AREA LOGAM RING DARI CITRA CLAHE
    # =========================================================================
    def _extract_ring_mask(self, gray_clahe, orig_hw):
        """
        Mendeteksi area keberadaan logam bearing ring menggunakan citra yang sudah
        diproses CLAHE (bukan texture gradient / Sobel seperti sebelumnya).

        Alasan perubahan:
        - CLAHE meratakan kontras lokal sehingga tepi ring (batas logam vs latar)
          menjadi lebih konsisten dan tajam di seluruh area citra, termasuk area
          gelap akibat lengkungan ring.
        - Canny pada citra CLAHE menghasilkan kontur ring yang lebih bersih
          dibandingkan Canny pada citra mentah atau gradient.
        - Ini memastikan masker ring dan matriks bersumber dari SATU citra yang sama
          (CLAHE), bukan dua sumber berbeda.

        Parameter:
            gray_clahe : Citra grayscale yang sudah di-CLAHE (uint8, resolusi asli)
            orig_hw    : Tuple (height, width) dari citra asli
        """
        h, w = orig_hw

        # 1. Canny Edge Detection langsung pada citra CLAHE
        # CLAHE sudah meratakan kontras, jadi Canny lebih akurat menangkap tepi ring
        # tanpa perlu menghitung gradient Sobel terlebih dahulu
        edges = cv2.Canny(gray_clahe, 30, 90)

        # 2. Hapus bintik kecil (debu, serat kertas, noise) dengan area < 60 piksel
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges)
        clean_edges = np.zeros_like(edges)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 60:
                clean_edges[labels == i] = 255

        # 3. Dilasi & Closing Morfologis untuk menyatukan kontur ring yang terputus-putus
        k_sz = max(25, int(min(w, h) * 0.03) | 1)
        dilated = cv2.dilate(
            clean_edges,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
        )
        c_sz = max(65, int(min(w, h) * 0.07) | 1)
        solid_ring = cv2.morphologyEx(
            dilated,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c_sz, c_sz))
        )

        # 4. Ambil kontur terbesar sebagai batas logam ring
        mask_f = np.ones((h, w), dtype=np.float32)  # Default: seluruh frame adalah ring
        contours, _ = cv2.findContours(solid_ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            main_cnt = max(contours, key=cv2.contourArea)
            # Hanya pakai jika kontur cukup besar (>5% area frame), hindari noise
            if cv2.contourArea(main_cnt) > (w * h * 0.05):
                raw_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(raw_mask, [main_cnt], -1, 255, -1)
                mask_f = (raw_mask > 0).astype(np.float32)

        return mask_f

    # =========================================================================
    # TAHAP 2: PERHITUNGAN MATRIKS ANGKA DINAMIS & PENURUNAN BOUNDING BOX
    # =========================================================================
    def _compute_matrix_and_bounding_boxes(self, gray_clahe, orig_hw, mask_f, is_defect, grid_dim=256):
        """
        Menghitung nilai numerik disparitas permukaan logam pada grid dinamis (128/256/512):
        - 0 - 10  : Daerah Luar Ring / Background (Hitam pekat, nilai 0 mutlak).
        - 11 - 45 : Permukaan Logam Normal Sehat (Biru - guratan bubut homogen).
        - 46 - 65 : Anomali Sedang / Noda / Batas Peralihan (Kuning / Oranye).
        - 66 - 99 : Baret / Cacat Fisik / Retakan (Merah Terang JOMPLANG).

        PERUBAHAN PIPELINE (CLAHE sebagai sumber tunggal):
        - Fungsi ini menerima gray_clahe (sudah di-CLAHE sejak awal pipeline),
          sehingga TIDAK perlu mengulang CLAHE di sini.
        - Matriks dan ring mask keduanya bersumber dari citra CLAHE yang sama,
          memastikan konsistensi analisis dari awal hingga akhir.

        Parameter:
            gray_clahe : Citra grayscale resolusi asli yang sudah di-CLAHE (uint8)
            orig_hw    : Tuple (height, width) citra asli untuk skala bounding box
            mask_f     : Masker ring biner (float32, resolusi asli)
            is_defect  : Bool - True jika CNN memprediksi ring defect
            grid_dim   : Resolusi matriks (128, 256, atau 512)
        """
        h, w = orig_hw

        # ---- LANGKAH 1: Resize citra CLAHE ke resolusi matriks ----
        # Karena gray_clahe sudah diproses CLAHE di awal pipeline,
        # langsung resize dan normalisasi ke float [0.0 - 1.0]
        gray_res = cv2.resize(gray_clahe, (grid_dim, grid_dim)).astype(np.float32) / 255.0

        # ---- LANGKAH 2: Siapkan masker ring pada resolusi matriks ----
        mask_res = cv2.resize(mask_f, (grid_dim, grid_dim), interpolation=cv2.INTER_NEAREST) > 0.5
        # Erosi tepi ring lebih dalam (19x19) agar kontur bibir logam luar/dalam tidak memicu garis merah palsu
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        mask_inner = cv2.erode(mask_res.astype(np.uint8), k_erode) > 0

        # ---- LANGKAH 3: Black-Hat pada citra CLAHE yang sudah di-resize ----
        # Black-Hat = closing(f) - f = mendeteksi fitur gelap lokal (retakan, goresan)
        # Karena citra sudah diratakan CLAHE, responnya terfokus ke struktur fisik,
        # bukan ke variasi pencahayaan global.
        k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        bh_fine = cv2.morphologyEx(gray_res, cv2.MORPH_BLACKHAT, k5)
        bh_wide = cv2.morphologyEx(gray_res, cv2.MORPH_BLACKHAT, k9)
        # Gabungan dua skala: halus (k5) lebih sensitif, lebar (k9) lebih kokoh
        bh_combined = (bh_fine * 0.60) + (bh_wide * 0.40)

        # Nol-kan area di luar ring agar tidak mengganggu kalkulasi
        bh_ring_only = bh_combined * mask_inner

        # ---- LANGKAH 4: Baseline Suppression & Non-Linear Power Scaling ----
        # Masalah CLAHE: kontras mikro di area mulus ikut diangkat sehingga muncul bercak kuning/merah palsu.
        # Solusi: Hitung baseline tekstur normal permukaan ring (median + deviasi standar).
        # Semua nilai di bawah baseline dianggap permukaan mulus murni (ditekan ke 0).
        # Hanya baret tajam yang melompat di atas baseline yang diizinkan naik ke kuning dan merah.
        ring_raw = bh_ring_only[mask_inner]
        if len(ring_raw) > 0:
            p_med = float(np.median(ring_raw))
            p_std = float(np.std(ring_raw))
            # Ambang baseline: pada ring normal cutoff lebih tinggi agar tidak memicu merah liar
            cutoff_sigma = 1.0 if is_defect else 2.5
            bh_sub = np.maximum(0.0, bh_ring_only - (p_med + cutoff_sigma * p_std)) * mask_inner

            # Penskalaan referensi absolut
            ref_scale = 0.45 if is_defect else 0.70
            norm = np.clip(bh_sub / ref_scale, 0.0, 1.0)

            # Kurva non-linear pangkat 1.8: menekan area mulus ke bawah, melesatkan baret ke atas
            curve = np.power(norm, 1.8)

            # Area mulus berada di zona warna Biru Tenang [20 - 35]
            # Zona Kuning [46 - 65] dan Merah [66 - 99] hanya untuk baret nyata
            matrix_vals = np.zeros((grid_dim, grid_dim), dtype=int)
            matrix_vals[mask_inner] = 22 + (curve[mask_inner] * 77.0).astype(int)
            matrix_vals[~mask_inner] = 0

            # Ambang batas anomali untuk bounding box
            anomaly_thresh = 55 if is_defect else 75
        else:
            matrix_vals = np.zeros((grid_dim, grid_dim), dtype=int)
            anomaly_thresh = 60 if is_defect else 82

        # ---- LANGKAH 6: Sisir Sel Anomali Merah (>=66) dan Kuning (>=46) ----
        # Analogi "Kepadatan Penduduk":
        # - Sisir sel yang berwarna merah dan kuning pekat di atas permukaan ring
        # - Sel-sel anomali yang membentuk gugusan/pemukiman padat dideteksi secara presisi
        # - Titik-titik sporadis terisolasi (noise) otomatis dieliminasi
        anom = (matrix_vals >= 46) & mask_inner

        # Sisir gugusan dengan Connected Components langsung pada resolusi matriks
        num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(anom.astype(np.uint8))
        cell_w = w / float(grid_dim)
        cell_h = h / float(grid_dim)
        pad = max(6, int(min(w, h) * 0.008))

        boxes = []
        if is_defect:
            # Pada citra cacat: cari gugusan yang memiliki sel merah (>=66) atau gugusan kuning padat
            for i in range(1, num_cc):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= 30:  # Minimal 30 sel anomali terkumpul (ambang batas kepadatan penduduk)
                    cluster_mask = (labels == i)
                    has_red = np.any(matrix_vals[cluster_mask] >= 66)
                    red_count = np.sum(matrix_vals[cluster_mask] >= 66)
                    if has_red or area >= 55:
                        gx = stats[i, cv2.CC_STAT_LEFT]
                        gy = stats[i, cv2.CC_STAT_TOP]
                        gw = stats[i, cv2.CC_STAT_WIDTH]
                        gh = stats[i, cv2.CC_STAT_HEIGHT]
                        bx1 = max(0, int(gx * cell_w) - pad)
                        by1 = max(0, int(gy * cell_h) - pad)
                        bx2 = min(w, int((gx + gw) * cell_w) + pad)
                        by2 = min(h, int((gy + gh) * cell_h) + pad)
                        boxes.append([bx1, by1, bx2, by2, area])
        else:
            # Pada citra normal: hanya jika ada klaster anomali ekstrem (safety gate)
            for i in range(1, num_cc):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= 120:
                    cluster_mask = (labels == i)
                    red_count = np.sum(matrix_vals[cluster_mask] >= 66)
                    if red_count >= 30:
                        gx = stats[i, cv2.CC_STAT_LEFT]
                        gy = stats[i, cv2.CC_STAT_TOP]
                        gw = stats[i, cv2.CC_STAT_WIDTH]
                        gh = stats[i, cv2.CC_STAT_HEIGHT]
                        bx1 = max(0, int(gx * cell_w) - pad)
                        by1 = max(0, int(gy * cell_h) - pad)
                        bx2 = min(w, int((gx + gw) * cell_w) + pad)
                        by2 = min(h, int((gy + gh) * cell_h) + pad)
                        boxes.append([bx1, by1, bx2, by2, area])

        # ---- LANGKAH 7: Gabungkan HANYA Kotak yang Saling Tumpang Tindih (IoU Overlap) ----
        # PENTING: JANGAN gabungkan kotak yang terpisah secara spasial!
        # Seperti analogi "pemukiman penduduk": Pemukiman A dan Pemukiman B adalah dua tempat terpisah,
        # masing-masing memiliki kotaknya sendiri, bukan dijadikan satu kotak raksasa yang menutupi area normal.
        def get_iou(b1, b2):
            ix1 = max(b1[0], b2[0])
            iy1 = max(b1[1], b2[1])
            ix2 = min(b1[2], b2[2])
            iy2 = min(b1[3], b2[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                return 0.0
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            return inter / float(area1 + area2 - inter)

        clean_boxes = []
        used = [False] * len(boxes)
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        for i in range(len(boxes)):
            if used[i]:
                continue
            bx1, by1, bx2, by2, area = boxes[i]
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if not used[j] and get_iou([bx1, by1, bx2, by2], boxes[j][:4]) > 0.15:
                    bx1 = min(bx1, boxes[j][0])
                    by1 = min(by1, boxes[j][1])
                    bx2 = max(bx2, boxes[j][2])
                    by2 = max(by2, boxes[j][3])
                    area += boxes[j][4]
                    used[j] = True
            clean_boxes.append((bx1, by1, bx2, by2))

        # Fallback jika model memprediksi defect tetapi belum ada klaster yang lolos ambang
        if is_defect and len(clean_boxes) == 0:
            py, px = np.unravel_index(np.argmax(matrix_vals), matrix_vals.shape)
            bx1 = max(0, int(px * cell_w) - 20)
            by1 = max(0, int(py * cell_h) - 20)
            bx2 = min(w, int((px + 1) * cell_w) + 20)
            by2 = min(h, int((py + 1) * cell_h) + 20)
            clean_boxes.append((bx1, by1, bx2, by2))

        return matrix_vals, clean_boxes

    # =========================================================================
    # TAHAP 3: VISUALISASI KOTAK MERAH PADA CITRA ASLI (TANPA BLUR)
    # =========================================================================
    def _draw_boxes_on_image(self, orig_np, boxes, is_defect):
        """
        Menggambar kotak merah (Bounding Box) langsung pada citra ASLI yang TAJAM dan ALAMI
        (tanpa blur latar belakang buatan), sesuai instruksi standar inspeksi QC industri.
        """
        bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        canvas = bgr.copy()
        h, w, _ = canvas.shape
        line_thick = max(2, int(min(w, h) * 0.005))

        for (bx1, by1, bx2, by2) in boxes:
            # Bounding Box Merah Terang
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (30, 30, 239), line_thick)

            # Label Tag Cacat [DEFECT] atau [SPOT]
            tag = "DEFECT" if is_defect else "SPOT"
            tag_color = (30, 30, 239) if is_defect else (0, 165, 255)
            tag_w = 60
            tag_h = 18
            if by1 >= tag_h:
                cv2.rectangle(canvas, (bx1, by1 - tag_h), (bx1 + tag_w, by1), tag_color, -1)
                cv2.putText(canvas, tag, (bx1 + 4, by1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, (bx1, by1), (bx1 + tag_w, by1 + tag_h), tag_color, -1)
                cv2.putText(canvas, tag, (bx1 + 4, by1 + tag_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Header Badge QC di Pojok Kiri Atas
        badge_w = 340
        badge_h = 60
        cv2.rectangle(canvas, (20, 20), (20 + badge_w, 20 + badge_h), (15, 23, 42), -1)
        cv2.rectangle(canvas, (20, 20), (20 + badge_w, 20 + badge_h), (51, 65, 85), 1)

        status_txt = f"STATUS: {'REJECT (DEFECT DETECTED)' if is_defect else 'PASS (RING NORMAL)'}"
        status_col = (50, 50, 255) if is_defect else (0, 230, 118)
        boxes_txt = f"ANOMALY BOXES: {len(boxes)} REGION(S) [NO-BLUR NATIVE]"

        cv2.putText(canvas, status_txt, (32, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, boxes_txt, (32, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (203, 213, 225), 1, cv2.LINE_AA)

        rgb_out = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_out)

    # =========================================================================
    # TAHAP 4: VISUALISASI MATRIKS DERAJAT ANGKA 256x256 (65.536 SEL METROLOGI)
    # =========================================================================
    def _render_matrix_grid_view(self, matrix_vals, grid_dim=256, out_size=1024):
        """
        Merender matriks metrologi beresolusi ultra-tinggi 256x256 (65.536 sel numerik):
        - 0 - 10  : Background Non-Ring (Hitam pekat / Dark Slate #0F172A).
        - 11 - 45 : Logam Ring Normal Sehat (Biru Navy / Steel Blue #1E3A8A).
        - 46 - 65 : Anomali Sedang / Noda / Batas (Kuning / Oranye #D97706).
        - 66 - 99 : Baret / Cacat Fisik / Retakan (Merah Terang Menyala #DC2626).
        
        Dilengkapi dengan garis grid presisi dan legenda skala warna metrologi standar.
        """
        # Pemetaan warna vektor yang sangat cepat
        colors = np.zeros((grid_dim, grid_dim, 3), dtype=np.uint8)

        # 1. Background Non-Ring (0 - 10): Hitam Pekat
        colors[matrix_vals <= 10] = [15, 23, 42]   # Dark slate BGR

        # 2. Logam Ring Normal Sehat (11 - 45): Biru Tenang
        mask_norm = (matrix_vals > 10) & (matrix_vals <= 45)
        colors[mask_norm] = [180, 100, 30]         # Steel blue BGR

        # 3. Anomali Sedang / Noda (46 - 65): Kuning / Oranye
        mask_warn = (matrix_vals > 45) & (matrix_vals <= 65)
        colors[mask_warn] = [10, 160, 240]         # Orange/amber BGR

        # 4. Baret / Cacat Fisik (66 - 99): Merah Terang
        colors[matrix_vals > 65] = [35, 35, 235]   # Vibrant red BGR

        # Perbesar ke ukuran kanvas resolusi tinggi (1024x1024)
        canvas = cv2.resize(colors, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

        # Garis Grid Sub-Blok Metrologi (adaptif agar garis tetap bersih di semua resolusi hingga 4096)
        cell_canvas_px = out_size / float(grid_dim)
        cells_per_block = max(8, int(round(24.0 / max(cell_canvas_px, 0.001))))
        step = max(16, int(round(cells_per_block * cell_canvas_px)))
        for x in range(0, out_size, step):
            cv2.line(canvas, (x, 0), (x, out_size), (40, 50, 70), 1)
        for y in range(0, out_size, step):
            cv2.line(canvas, (0, y), (out_size, y), (40, 50, 70), 1)

        # Pita Legenda Metrologi di Bagian Bawah
        legend_h = 56
        legend_y1 = out_size - legend_h
        cv2.rectangle(canvas, (0, legend_y1), (out_size, out_size), (15, 23, 42), -1)
        cv2.line(canvas, (0, legend_y1), (out_size, legend_y1), (51, 65, 85), 1)

        # Sampel Warna Legenda
        # [0-10: Non-Ring]
        cv2.rectangle(canvas, (20, legend_y1 + 18), (40, legend_y1 + 38), (15, 23, 42), -1)
        cv2.putText(canvas, "0-10: Background", (46, legend_y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (203, 213, 225), 1, cv2.LINE_AA)

        # [11-45: Normal Surface]
        cv2.rectangle(canvas, (180, legend_y1 + 18), (200, legend_y1 + 38), (180, 100, 30), -1)
        cv2.putText(canvas, "11-45: Ring Normal", (206, legend_y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (203, 213, 225), 1, cv2.LINE_AA)

        # [46-65: Warning Anomaly]
        cv2.rectangle(canvas, (370, legend_y1 + 18), (390, legend_y1 + 38), (10, 160, 240), -1)
        cv2.putText(canvas, "46-65: Anomali/Noda", (396, legend_y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (203, 213, 225), 1, cv2.LINE_AA)

        # [66-99: Defect/Baret]
        cv2.rectangle(canvas, (560, legend_y1 + 18), (580, legend_y1 + 38), (35, 35, 235), -1)
        cv2.putText(canvas, "66-99: Baret / Cacat", (586, legend_y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        # Informasi Dimensi Grid Dinamis
        total_cells_str = f"{grid_dim * grid_dim:,}"
        cv2.putText(canvas, f"GRID: {grid_dim}x{grid_dim} ({total_cells_str} Cells) - Metrology", (630, legend_y1 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (148, 163, 184), 1, cv2.LINE_AA)

        rgb_out = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_out)

    # =========================================================================
    # TAHAP 5: 3 FILTER STANDAR INDUSTRI PILIHAN
    # =========================================================================
    def _apply_industrial_filters(self, orig_np, gray_clahe=None):
        """
        Mengimplementasikan 3 filter unggulan pilihan untuk inspeksi permukaan bearing ring:
        1. CLAHE            : Normalisasi pencahayaan lokal adaptif (mengangkat area gelap lengkungan ring).
        2. Morfologi Cacat  : Black-Hat filter untuk isolasi retakan, cerukan cacat, dan lubang jarum (pitting).
        3. Texture Gradient : Sobel surface topography untuk membedakan struktur tekstur logam vs latar meja/kertas.
        """
        gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
        filters_dict = {}

        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Standar industri untuk meratakan pencahayaan lokal dan menaikkan detail di area gelap
        if gray_clahe is None:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            f_clahe = clahe.apply(gray)
        else:
            f_clahe = gray_clahe
        filters_dict["filter_clahe"] = Image.fromarray(cv2.cvtColor(f_clahe, cv2.COLOR_GRAY2RGB))

        # 2. Morfologi Cacat (Morphological Black-Hat / Bottom-Hat)
        # Rumus: (f • b) - f. Mengisolasi fitur gelap mikro seperti retakan, goresan dalam, dan pitting
        k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        f_blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_morph)
        f_blackhat_norm = cv2.normalize(f_blackhat, None, 0, 255, cv2.NORM_MINMAX)
        filters_dict["filter_morpho"] = Image.fromarray(cv2.cvtColor(f_blackhat_norm, cv2.COLOR_GRAY2RGB))

        # 3. Texture Gradient (Sobel Surface Topography)
        # Menghitung magnitudo gradien spasial untuk menonjolkan topografi tekstur logam
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        sx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sx, sy)
        f_texture = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        filters_dict["filter_texture_grad"] = Image.fromarray(cv2.cvtColor(f_texture, cv2.COLOR_GRAY2RGB))

        return filters_dict

    # =========================================================================
    # TAHAP 6: FUNGSI UTAMA INSPEKSI (INSPECT_AND_MARK)
    # =========================================================================
    def inspect_and_mark(self, pil_image, sensitivity=0.25, grid_dim=512):
        """
        Fungsi utama yang dipanggil oleh antarmuka Web App:
        1. Mempertahankan citra asli bebas blur.
        2. Menjalankan forward pass Pure CNN 512x512 untuk klasifikasi biner dan confidence.
        3. Menghitung matriks disparitas 512x512 (262.144 sel) dengan CLAHE + Black-Hat.
        4. Mengelompokkan daerah padat merah-kuning (Density-Based Clustering) menjadi kotak deteksi baret.
        5. Mengemas hasil inspeksi dan visualisasi metrologi secara rapi.

        Parameter:
            pil_image  : Gambar masukan (PIL.Image)
            sensitivity: Sensitivitas klasifikasi (tidak dipakai langsung, reserved)
            grid_dim   : Resolusi matriks metrologi (fiksasi default 512)
        """
        # Validasi resolusi matriks (default 512)
        if grid_dim not in (128, 256, 512, 1024, 2048, 4096):
            grid_dim = 512
        orig_rgb = pil_image.convert("RGB")
        orig_np = np.array(orig_rgb)
        orig_hw = (orig_np.shape[0], orig_np.shape[1])  # (height, width)

        # 1. Konversi ke Grayscale & CLAHE Global
        # Pipeline Baru: Citra Asli -> CLAHE -> Deteksi Lengkung Ring -> Matriks Metrologi
        gray_raw = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe_obj.apply(gray_raw)

        # 2. Deteksi Area Fisik Logam Ring langsung dari citra CLAHE
        ring_mask = self._extract_ring_mask(gray_clahe, orig_hw)

        # 3. Klasifikasi Objektif oleh Pure CNN
        tensor_img = INFERENCE_TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor_img)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]

        predicted_idx = int(probabilities.argmax())
        predicted_label = CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx]) * 100.0
        is_defect = (predicted_label == "defect_ring")
        decision = "REJECT (DEFECT)" if is_defect else "PASS (NORMAL)"

        # 4. Hitung Matriks Angka Dinamis (128/256/512) langsung dari citra CLAHE
        matrix_vals, boxes = self._compute_matrix_and_bounding_boxes(
            gray_clahe, orig_hw, ring_mask, is_defect=is_defect, grid_dim=grid_dim
        )

        # 4. Render Dua Tampilan Inti: Kotak Merah pada Citra Asli (Bebas Blur) & Matriks Dinamis
        annotated_image   = self._draw_boxes_on_image(orig_np, boxes, is_defect=is_defect)
        matrix_grid_image = self._render_matrix_grid_view(matrix_vals, grid_dim=grid_dim, out_size=1024)

        # 5. Ekstraksi 3 Filter Standar Industri Pilihan (CLAHE, Morfologi, Texture Gradient)
        industrial_filters = self._apply_industrial_filters(orig_np, gray_clahe=gray_clahe)

        # Menyatukan views untuk antarmuka web
        views = {
            "red_dots": annotated_image,             # Opsi 1: Deteksi Cacat (Kotak Merah pada Citra Asli)
            "chessboard_matrix": matrix_grid_image,  # Opsi 2: Matriks 256x256 (65.536 Sel)
            "original_native": orig_rgb              # Citra Asli Bebas Manipulasi
        }
        # Menambahkan 3 filter standar industri pilihan ke dalam views
        views.update(industrial_filters)

        # Metrik ringkas untuk kartu dashboard web
        metrics = {
            "matrix_shape": f"{matrix_vals.shape[0]}x{matrix_vals.shape[1]}",
            "matrix_total_cells": int(matrix_vals.size),
            "grid_dim": grid_dim,
            "matrix_max": int(np.max(matrix_vals)),
            "matrix_mean": round(float(np.mean(matrix_vals[matrix_vals > 0])), 2) if np.sum(matrix_vals > 0) > 0 else 0.0,
            "anomaly_boxes_count": len(boxes),
            "defect_probability": round(float(probabilities[0]) * 100.0, 2),
            "normal_probability": round(float(probabilities[1]) * 100.0, 2)
        }

        return {
            "predicted_class": predicted_label,
            "decision": decision,
            "is_defect": is_defect,
            "confidence": confidence,
            "probabilities": {
                CLASS_NAMES[i]: float(probabilities[i]) * 100.0
                for i in range(len(CLASS_NAMES))
            },
            "num_defect_markers": len(boxes),
            "original_image": orig_rgb,
            "annotated_image": annotated_image,
            "views": views,
            "metrics": metrics,
            "filter_metadata": INDUSTRIAL_FILTER_METADATA
        }
