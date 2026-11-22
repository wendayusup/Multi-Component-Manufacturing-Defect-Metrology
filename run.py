"""
UNIFIED VISUAL INSPECTION APPLICATION LAUNCHER
----------------------------------------------
POLITEKNIK MANUFAKTUR BANDUNG (POLMAN)
TUGAS BESAR ARTIFICIAL INTELLIGENCE & METROLOGY

Integrates 3 modules into one web interface:
1. Bearing Ring Inspection (RingClassification)
2. Casting / Bearing Inspection (Casting_Detection)
3. Plastic Defect Inspection (Standby)
"""
import sys
from pathlib import Path

# Ensure application root and workspace root are in sys.path
INTEGRATION_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = INTEGRATION_ROOT.parent

for p in [str(INTEGRATION_ROOT), str(WORKSPACE_ROOT / "Casting_Detection"), str(WORKSPACE_ROOT / "RingClassification"), str(WORKSPACE_ROOT / "PlasticDefectionWeb")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from unified_app import create_app
from unified_app.config import HOST, PORT, DEBUG

app = create_app()

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("      UNIFIED VISUAL INSPECTION STATION (POLMAN BANDUNG)")
    print("=" * 70)
    print(f"   Local URL   : http://127.0.0.1:{PORT}")
    print(f"   Network URL : http://{HOST}:{PORT}")
    print("   Active Modules:")
    print("     [1] Bearing Ring Inspection    : READY (Pure CNN 512x512 Metrology)")
    print("     [2] Casting/Bearing Inspection : READY (Pure-NumPy CNN & Grad-CAM)")
    print("     [3] Plastic Defect Inspection  : READY (PyTorch Single-Class FPN CNN)")
    print("=" * 70 + "\n")
    app.run(host=HOST, port=PORT, debug=DEBUG)
