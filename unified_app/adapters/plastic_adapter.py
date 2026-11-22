"""
PLASTIC DEFECT INSPECTION ADAPTER (STANDBY / SCAFFOLD)
------------------------------------------------------
Scaffold adapter for the upcoming Plastic Defect Detection module:
- Preserves clean system interface
- Reports status as 'standby_unavailable' until folder and model weights
  are delivered by teammate.
- Once folder is supplied, inference logic can be plugged in directly
  without changing the unified web application.
"""
from typing import Dict, Any
from .base import BaseInspectionAdapter


class PlasticAdapter(BaseInspectionAdapter):
    module_id = "plastic"
    display_name = "Plastic Defect Inspection"
    category = "Plastic Injection"

    def __init__(self):
        self.is_ready = False
        self.load_error = "Folder model plastic belum diserahkan oleh rekan tim. Modul berada dalam status standby."

    def load(self) -> None:
        pass

    def health(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "display_name": self.display_name,
            "category": self.category,
            "status": "unavailable",
            "message": "Menunggu Penyerahan Folder/Bobot Model dari Rekan Tim (Standby Integration)",
            "error_message": self.load_error,
            "capabilities": ["classification", "defect_localization"]
        }

    def inspect(self, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "display_name": self.display_name,
            "status": "model_unavailable",
            "decision": "STANDBY",
            "predicted_label": "Modul Belum Tersedia",
            "confidence": 0.0,
            "message": "Modul Plastic Defect Inspection sedang menunggu penyerahan file model dari rekan tim. Silakan gunakan tab Bearing atau Ring terlebih dahulu."
        }
