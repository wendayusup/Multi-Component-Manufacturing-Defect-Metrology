"""
CENTRAL INTEGRATION CONFIGURATION
---------------------------------
Global settings for Unified Inspection Web Application.
"""
from pathlib import Path

# Paths
APP_ROOT = Path(__file__).resolve().parent
INTEGRATION_ROOT = APP_ROOT.parent
WORKSPACE_ROOT = INTEGRATION_ROOT.parent

# Model Development Directories (Self-contained inside Unified_Inspection)
MODEL_DEV_ROOT = INTEGRATION_ROOT / "model_development"
RING_DIR = MODEL_DEV_ROOT / "ring_inspection"
CASTING_DIR = MODEL_DEV_ROOT / "bearing_inspection"
PLASTIC_DIR = MODEL_DEV_ROOT / "plastic_inspection"

# Fallbacks for flexibility
if not CASTING_DIR.exists():
    CASTING_DIR = MODEL_DEV_ROOT / "bearing_casting"
if not RING_DIR.exists():
    RING_DIR = WORKSPACE_ROOT / "RingClassification"
if not CASTING_DIR.exists():
    CASTING_DIR = WORKSPACE_ROOT / "Casting_Detection"

# Web Server
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False

# Limits
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}

# Module Metadata
MODULES = {
    "ring": {
        "id": "ring",
        "name": "Bearing Ring Inspection",
        "category": "Bearing Surface",
        "badge": "Pure CNN 512x512",
        "description": "Inspeksi cacat baret pada cincin logam bearing dengan matriks disparitas 512x512 dan klasterisasi connected components."
    },
    "bearing": {
        "id": "bearing",
        "name": "Casting / Bearing Inspection",
        "category": "Casting Metal",
        "badge": "Pure-NumPy CNN",
        "description": "Pemeriksaan cacat coran logam (casting) dengan Pure-NumPy CNN, gerbang kualitas, dan lokalisasi Grad-CAM."
    },
    "plastic": {
        "id": "plastic",
        "name": "Plastic Defect Inspection",
        "category": "Plastic Injection",
        "badge": "Single-Class CNN (FPN)",
        "description": "Inspeksi cacat produk cetakan plastik injection molding berbasis Feature Pyramid Single-Class CNN dengan dekoder NMS."
    }
}
