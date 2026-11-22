"""
TAHAP 2: ARSITEKTUR MODEL PURE CONVOLUTIONAL NEURAL NETWORK (CNN) 512x512
------------------------------------------------------------------------
Implementasi arsitektur 7 tahapan standar CNN murni (tanpa transfer learning):
1. Input Image: Citra Grayscale resolusi tinggi 512x512 (1 kanal).
2. Convolution Layer + Batch Normalization: Ekstraksi fitur mikro baret dan guratan permukaan ring.
3. ReLU Layer: Fungsi aktivasi non-linier f(x) = max(0, x) untuk memutus linieritas.
4. Pooling Layer: Downsampling bertahap menggunakan Max Pooling (2x2) untuk menjaga invarian spasial.
5. Flatten Layer: Mengonversi tensor fitur 3D menjadi vektor 1D terstruktur.
6. Fully Connected (Dense) Layer: Jaringan saraf klasifikasi dengan Dropout (0.4) pencegah overfitting.
7. Output Layer: Logits dipetakan ke probabilitas kelas melalui Softmax (defect_ring vs normal_ring).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PureCNN(nn.Module):
    """
    Model Pure CNN untuk klasifikasi permukaan bearing ring resolusi 512x512 piksel.
    Menggunakan 4 blok konvolusi berjenjang + Adaptive Pooling untuk menjaga stabilitas parameter.
    """
    def __init__(self, in_channels=1, num_classes=2):
        super(PureCNN, self).__init__()

        # -------------------------------------------------------------
        # BLOK 1: Ekstraksi Fitur Dasar (Tepian, Siluet Ring, Kontras Awal)
        # Input: (B, 1, 512, 512) -> Output: (B, 16, 256, 256)
        # -------------------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------------------------------------------------------------
        # BLOK 2: Ekstraksi Fitur Tekstur Menengah (Guratan Bubut, Lengkungan)
        # Input: (B, 16, 256, 256) -> Output: (B, 32, 128, 128)
        # -------------------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------------------------------------------------------------
        # BLOK 3: Ekstraksi Fitur Kompleks (Pola Baret, Retakan, Goresan)
        # Input: (B, 32, 128, 128) -> Output: (B, 64, 64, 64)
        # -------------------------------------------------------------
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------------------------------------------------------------
        # BLOK 4: Ekstraksi Fitur Semantik Tingkat Tinggi
        # Input: (B, 64, 64, 64) -> Output: (B, 64, 32, 32)
        # -------------------------------------------------------------
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------------------------------------------------------------
        # Tahap 4 (Lanjutan): Adaptive Pooling untuk Fiksasi Dimensi
        # Menyeragamkan ukuran peta fitur ke (8, 8) agar ukuran vektor dense stabil
        # Output: (B, 64, 8, 8)
        # -------------------------------------------------------------
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # -------------------------------------------------------------
        # Tahap 5: Flatten Layer
        # Mengubah tensor 3D (64 x 8 x 8 = 4.096 fitur) menjadi vektor 1D
        # -------------------------------------------------------------
        self.flatten = nn.Flatten()

        # -------------------------------------------------------------
        # Tahap 6: Fully Connected (Dense) Layer dengan Dropout
        # -------------------------------------------------------------
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)

        # -------------------------------------------------------------
        # Tahap 7: Output Classification Layer
        # Menghasilkan skor logits untuk 2 kelas: [defect_ring, normal_ring]
        # -------------------------------------------------------------
        self.fc_out = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        """
        Alur forward pass berurutan sesuai bagan tahapan algoritma CNN.
        """
        # Blok Konvolusi 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Blok Konvolusi 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Blok Konvolusi 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Blok Konvolusi 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # Adaptive Pooling & Flatten
        x = self.adaptive_pool(x)
        x = self.flatten(x)

        # Klasifikasi Dense
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)

        logits = self.fc_out(x)
        return logits

    def predict_proba(self, x):
        """
        Mengubah skor logits mentah menjadi distribusi probabilitas [0.0 - 1.0] dengan Softmax.
        """
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities


def inspect_architecture():
    """
    Fungsi verifikasi: simulasi forward pass dengan dummy input 512x512
    dan mencetak transformasi dimensi tiap tahap.
    """
    print("=== [TAHAP 2] VERIFIKASI ARSITEKTUR PURE CNN 512x512 ===")
    model = PureCNN(in_channels=1, num_classes=2)
    model.eval()

    dummy_input = torch.randn(1, 1, 512, 512)
    print(f"0. Citra Masukan (Input)      : {dummy_input.shape}")

    x = model.pool1(model.relu1(model.bn1(model.conv1(dummy_input))))
    print(f"1. Setelah Blok Conv1 (16)     : {x.shape}")

    x = model.pool2(model.relu2(model.bn2(model.conv2(x))))
    print(f"2. Setelah Blok Conv2 (32)     : {x.shape}")

    x = model.pool3(model.relu3(model.bn3(model.conv3(x))))
    print(f"3. Setelah Blok Conv3 (64)     : {x.shape}")

    x = model.pool4(model.relu4(model.bn4(model.conv4(x))))
    print(f"4. Setelah Blok Conv4 (64)     : {x.shape}")

    x = model.adaptive_pool(x)
    print(f"5. Setelah Adaptive Pool (8x8) : {x.shape}")

    x = model.flatten(x)
    print(f"6. Setelah Flatten (Vektor 1D) : {x.shape} (Total: {x.shape[1]} fitur)")

    x = model.dropout(model.relu_fc(model.fc1(x)))
    print(f"7. Setelah Dense Layer (128)   : {x.shape}")

    logits = model.fc_out(x)
    print(f"8. Output Logits (2 Kelas)     : {logits.shape}")

    probs = F.softmax(logits, dim=1)
    print(f"9. Probabilitas Softmax        : {probs.squeeze().tolist()} (Total = {probs.sum().item():.2f})")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n-> Total Bobot Terlatih        : {total_params:,} parameter")
    print("=== [TAHAP 2] VERIFIKASI SELESAI: Arsitektur Pure CNN 512x512 Siap Dilatih ===")


if __name__ == "__main__":
    inspect_architecture()
