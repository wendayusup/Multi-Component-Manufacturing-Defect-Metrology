"""
MODEL REGISTRY SERVICE
----------------------
Manages singleton instances of adapters and provides unified access.
"""
from typing import Dict, Any, Optional
from ..adapters import BearingAdapter, RingAdapter, PlasticAdapter, BaseInspectionAdapter


class ModelRegistry:
    def __init__(self):
        self._adapters: Dict[str, BaseInspectionAdapter] = {}
        self._initialize_adapters()

    def _initialize_adapters(self):
        # Ring Adapter
        try:
            self._adapters["ring"] = RingAdapter()
            print("[INFO] Model Registry: Ring Adapter initialized.")
        except Exception as e:
            print(f"[WARN] Model Registry: Ring Adapter failed: {e}")

        # Bearing / Casting Adapter
        try:
            self._adapters["bearing"] = BearingAdapter()
            print("[INFO] Model Registry: Bearing/Casting Adapter initialized.")
        except Exception as e:
            print(f"[WARN] Model Registry: Bearing/Casting Adapter failed: {e}")

        # Plastic Adapter
        try:
            self._adapters["plastic"] = PlasticAdapter()
            print("[INFO] Model Registry: Plastic Adapter initialized.")
        except Exception as e:
            print(f"[WARN] Model Registry: Plastic Adapter failed: {e}")

    def get_adapter(self, module_id: str) -> Optional[BaseInspectionAdapter]:
        return self._adapters.get(module_id)

    def get_all_health(self) -> Dict[str, Any]:
        return {
            m_id: adapter.health() for m_id, adapter in self._adapters.items()
        }


# Singleton registry instance
_registry_instance = None


def get_model_registry() -> ModelRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance
