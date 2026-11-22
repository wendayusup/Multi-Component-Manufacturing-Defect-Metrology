"""
TAHAP 1: PERSIAPAN DAN PEMUATAN DATASET CITRA 512x512 DENGAN FILTER CLAHE
------------------------------------------------------------------------
Modul ini menangani pipeline prapemrosesan dataset bearing ring:
1. Mengubah citra ke format Grayscale (1 kanal).
2. Menerapkan filter standar industri CLAHE (Contrast Limited Adaptive Histogram Equalization)
   agar model CNN belajar langsung dari fitur permukaan yang telah dinormalkan pencahayaannya.
3. Menyeragamkan ukuran citra ke resolusi standar 512x512 piksel.
4. Mengonversi citra menjadi Tensor PyTorch dengan rentang nilai [0.0, 1.0].
5. Membagi dataset ke dalam DataLoader (Train, Validation, Test) secara terstruktur.
"""

import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Direktori utama proyek
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_SPLIT_DIR = BASE_DIR / "dataset_split"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ApplyCLAHE(object):
    """
    Transformasi kustom PyTorch untuk mengaplikasikan filter CLAHE
    pada citra PIL grayscale sebelum dikonversi menjadi tensor.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        # Konversi PIL Image Grayscale ke array NumPy uint8
        np_img = np.array(img)
        # Terapkan ekualisasi histogram adaptif CLAHE
        eq_img = self.clahe.apply(np_img)
        # Kembalikan sebagai PIL Image
        return Image.fromarray(eq_img)


def get_data_transforms(image_size=(512, 512)):
    """
    Pipeline transformasi citra masukan model CNN 512x512:
    1. Grayscale: konversi citra menjadi 1 kanal intensitas.
    2. CLAHE: normalisasi pencahayaan lokal adaptif.
    3. Resize: standarisasi dimensi ke resolusi 512x512 piksel.
    4. ToTensor: konversi ke tensor PyTorch dengan normalisasi [0.0, 1.0].
    5. Normalize: normalisasi mean=0.5 dan std=0.5 untuk stabilitas konvergensi gradien.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform


def get_data_loaders(batch_size=16, image_size=(512, 512)):
    """
    Membuat DataLoader PyTorch untuk subset Train, Validation, dan Test.
    """
    transform = get_data_transforms(image_size)

    train_path = os.path.join(DATASET_SPLIT_DIR, "train")
    val_path   = os.path.join(DATASET_SPLIT_DIR, "val")
    test_path  = os.path.join(DATASET_SPLIT_DIR, "test")

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset   = datasets.ImageFolder(root=val_path,   transform=transform)
    test_dataset  = datasets.ImageFolder(root=test_path,  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    classes = train_dataset.classes  # ['defect_ring', 'normal_ring']

    return train_loader, val_loader, test_loader, classes


def verify_and_preview():
    """
    Fungsi pengujian data loader untuk memastikan ukuran tensor 512x512 dan statistik normalisasi.
    """
    print("=== [TAHAP 1] VERIFIKASI DATA LOADER RESOLUSI 512x512 ===")
    train_loader, val_loader, test_loader, classes = get_data_loaders(batch_size=16, image_size=(512, 512))

    print(f"Kelas terdeteksi: {classes}")
    print(f"Jumlah Train : {len(train_loader.dataset)} citra ({len(train_loader)} batch)")
    print(f"Jumlah Val   : {len(val_loader.dataset)} citra ({len(val_loader)} batch)")
    print(f"Jumlah Test  : {len(test_loader.dataset)} citra ({len(test_loader)} batch)")

    images, labels = next(iter(train_loader))
    print(f"\nBentuk tensor batch: {images.shape} (Batch, Channel, Height, Width)")
    print(f"Rentang nilai piksel: min = {images.min().item():.3f}, max = {images.max().item():.3f}")
    print("=== [TAHAP 1] VERIFIKASI SELESAI: DataLoader 512x512 siap digunakan ===")


if __name__ == "__main__":
    verify_and_preview()
