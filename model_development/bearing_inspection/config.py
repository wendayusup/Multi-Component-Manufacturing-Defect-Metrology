"""
Application and Model Configuration for Casting Defect Detector.
Provides centralized parameters for file validation, image preprocessing,
quality gates, CNN model settings, and server options.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("DATASET_DIR", BASE_DIR / "dataset"))
WEIGHTS_DIR = BASE_DIR / "weights"
DATA_DIR = BASE_DIR / "data"

# Web & Upload Settings
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB limit
ENABLE_IMAGE_PERSISTENCE = False
UPLOAD_TMP_DIR = DATA_DIR / "uploads"

# Image Preprocessing Settings
IMAGE_TARGET_SIZE = (64, 64)  # (width, height)
IMAGE_CHANNELS = 1            # 1 for Grayscale
NORMALIZATION_SCALE = 255.0

# Image Quality Check Thresholds
QUALITY_MIN_WIDTH = 100
QUALITY_MIN_HEIGHT = 100
QUALITY_MIN_BRIGHTNESS = 25.0   # On 0-255 scale
QUALITY_MAX_BRIGHTNESS = 235.0  # On 0-255 scale
QUALITY_BLUR_THRESHOLD = 15.0   # Laplacian variance proxy

# Custom CNN & Decision Settings
CLASS_NAMES = ["OK", "Defective"]
MODEL_VERSION = "casting-cnn-0.1.0"
DEFAULT_WEIGHTS_FILE = WEIGHTS_DIR / "casting_cnn_v1.npz"
DEFAULT_META_FILE = WEIGHTS_DIR / "casting_cnn_v1_meta.json"
CONFIDENCE_THRESHOLD = 0.70  # Below this, status is 'review_required'

# Database Settings
DATABASE_PATH = DATA_DIR / "inspections.db"

# Server Settings
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

