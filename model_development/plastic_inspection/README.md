# Web inspeksi plastik

Aplikasi Flask untuk checkpoint PlasticDefection_Detector.ipynb. Operator mengunggah foto asli (termasuk 1920 × 1080). Modul dari notebook menyesuaikan input, mendekode kotak, melakukan NMS, dan menggambar hasil pada koordinat foto yang sudah dikoreksi orientasinya. File upload diproses di memori.

## Pasang hasil Colab

1. Jalankan notebook hingga bagian 7. Buka folder run yang diinginkan di MyDrive/PlasticDefection/runs dan unduh plastic_web_model.zip.
2. Ekstrak **isi** ZIP model ke folder models/ di sebelah app.py. Hasilnya harus berupa models/plastic_defect_detector.pt, models/detector.py dan models/environment.json. Gunakan detector.py dari run yang sama. Versi yang sesuai dengan notebook terlampir sudah disertakan.
3. Periksa environment.json bila perlu menyamakan versi dengan Colab. requirements.txt menggunakan pasangan versi PyTorch dan torchvision untuk Python 3.11. Pilih wheel CPU untuk menjalankan inferensi tanpa GPU.

## PyCharm di Windows

Ekstrak ZIP aplikasi dan buka folder PlasticDefectionWeb dengan Open Folder. Di terminal PowerShell folder tersebut jalankan:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Buka http://127.0.0.1:5000. Pilih foto JPG/PNG dan tekan Analisis foto. Hasil memuat kotak, confidence, koordinat dalam piksel, waktu proses dan tombol unduh. Uji dengan satu foto NOK dan satu foto OK dari run Colab yang sama untuk membandingkan hasil. Jika perlu meninjau detail kecil, unduh hasil resolusi penuh.

## PythonAnywhere

Upload folder aplikasi ke /home/USERNAME/PlasticDefectionWeb termasuk checkpoint. Dalam Bash console, buat virtualenv menggunakan versi Python yang juga dipilih pada tab Web. Contoh untuk Python 3.11:

```bash
cd ~/PlasticDefectionWeb
mkvirtualenv --python=python3.11 plasticweb
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -c "from app import load_bundle; print(load_bundle()[1:])"
```

Pilih Web → Add a new web app → Manual configuration. Isi Virtualenv dengan /home/USERNAME/.virtualenvs/plasticweb. Pada file WSGI yang ditautkan dari tab Web, gunakan kode berikut dengan username akun sendiri:

```python
import sys
sys.path.insert(0, '/home/USERNAME/PlasticDefectionWeb')
from app import app as application
```

Klik Reload. Uji upload lewat URL publik. Instalasi memerlukan akses ke wheel dan kuota disk yang cukup untuk PyTorch. Jika impor gagal, cek Error log pada tab Web serta interpreter virtualenv. Jika checkpoint diganti, reload aplikasi untuk memuat bobot dan ambang yang baru.

Panduan hosting: https://help.pythonanywhere.com/pages/Flask/
