"""
Example Configuration for Casting Defect Detector.
Copy to config.py or set corresponding environment variables.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("DATASET_DIR", BASE_DIR / "dataset"))
WEIGHTS_DIR = BASE_DIR / "weights"
DATA_DIR = BASE_DIR / "data"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

ENABLE_IMAGE_PERSISTENCE = False
UPLOAD_TMP_DIR = DATA_DIR / "uploads"

IMAGE_TARGET_SIZE = (64, 64)
IMAGE_CHANNELS = 1
NORMALIZATION_SCALE = 255.0

QUALITY_MIN_WIDTH = 100
QUALITY_MIN_HEIGHT = 100
QUALITY_MIN_BRIGHTNESS = 25.0
QUALITY_MAX_BRIGHTNESS = 235.0
QUALITY_BLUR_THRESHOLD = 15.0

CLASS_NAMES = ["OK", "Defective"]
MODEL_VERSION = "casting-cnn-0.1.0"
DEFAULT_WEIGHTS_FILE = WEIGHTS_DIR / "casting_cnn_v1.npz"
DEFAULT_META_FILE = WEIGHTS_DIR / "casting_cnn_v1_meta.json"
CONFIDENCE_THRESHOLD = 0.70

DATABASE_PATH = DATA_DIR / "inspections.db"
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False

