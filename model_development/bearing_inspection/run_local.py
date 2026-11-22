"""
Local Execution Entry Point for Casting Defect Detector.
Launches the Flask web server with configured host, port, and debug options.
"""
import sys
from pathlib import Path

# Ensure project root is in Python sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from app import create_app

app = create_app()

if __name__ == "__main__":
    print("=" * 65)
    print("  LAUNCHING CASTING DEFECT DETECTOR (MVP)")
    print("=" * 65)
    print(f"Server URL    : http://127.0.0.1:{config.PORT}")
    print(f"Network URL   : http://{config.HOST}:{config.PORT}")
    print(f"Model Version : {config.MODEL_VERSION}")
    print(f"Max Upload    : {int(config.MAX_CONTENT_LENGTH / (1024 * 1024))} MB")
    print("=" * 65)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)

