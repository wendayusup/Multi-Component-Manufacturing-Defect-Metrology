"""
BASE INSPECTION ADAPTER CONTRACT
--------------------------------
Defines standard interface for all inspection modules in Unified Inspection System.
Each module implements load(), health(), and inspect() independently.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseInspectionAdapter(ABC):
    module_id: str
    display_name: str
    category: str

    @abstractmethod
    def load(self) -> None:
        """Loads and caches the model weights into memory."""
        pass

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Returns readiness state, model metadata, and health details."""
        pass

    @abstractmethod
    def inspect(self, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Runs domain-specific preprocessing, inference, and postprocessing."""
        pass
