"""
Offline Dataset Loader for Casting Product Inspection.
Discovers and loads casting images from the local dataset/ directory.
Supports reproducible train/val/test splitting, image caching/loading, and label mapping.
"""
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

import config


# Standard label mapping for casting product folders
DEFAULT_LABEL_MAPPING = {
    "ok_front": "OK",
    "def_front": "Defective",
    "ok": "OK",
    "def": "Defective",
    "defective": "Defective"
}


class CastingDatasetLoader:
    """
    Offline dataset loader for casting images.
    Reads images from the configured dataset folder, validates readability,
    and formats data for model training and evaluation.
    """
    def __init__(
        self,
        dataset_dir: Optional[Path] = None,
        target_size: Tuple[int, int] = config.IMAGE_TARGET_SIZE,
        label_mapping: Optional[Dict[str, str]] = None
    ):
        self.dataset_dir = Path(dataset_dir or config.DATASET_DIR)
        self.target_size = target_size
        self.label_mapping = label_mapping or DEFAULT_LABEL_MAPPING
        self.class_names = config.CLASS_NAMES  # ["OK", "Defective"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def discover_images(self, sub_path: Optional[str] = None) -> List[Dict]:
        """
        Recursively scans the dataset directory for supported images and resolves labels.
        Returns a list of dicts: {'path': Path, 'relative_path': str, 'label': str, 'class_idx': int}
        """
        search_root = self.dataset_dir / sub_path if sub_path else self.dataset_dir
        if not search_root.exists():
            return []

        discovered = []
        valid_extensions = {f".{ext}" for ext in config.ALLOWED_EXTENSIONS}

        for root, _, files in os.walk(search_root):
            root_path = Path(root)
            folder_name = root_path.name.lower()

            # Check if this folder matches known class names
            matched_label = self.label_mapping.get(folder_name)
            if not matched_label:
                # Check parent folder as well
                parent_name = root_path.parent.name.lower()
                matched_label = self.label_mapping.get(parent_name)

            if matched_label and matched_label in self.class_to_idx:
                for file_name in files:
                    ext = Path(file_name).suffix.lower()
                    if ext in valid_extensions:
                        file_path = root_path / file_name
                        try:
                            rel_path = file_path.relative_to(config.BASE_DIR).as_posix()
                        except ValueError:
                            rel_path = file_path.as_posix()

                        discovered.append({
                            "path": file_path,
                            "relative_path": rel_path,
                            "label": matched_label,
                            "class_idx": self.class_to_idx[matched_label]
                        })

        return discovered

    def audit(self) -> Dict:
        """
        Audits the dataset directory: checks total counts, class distribution,
        and tests opening sample images to verify integrity.
        """
        if not self.dataset_dir.exists():
            return {
                "dataset_dir": str(self.dataset_dir),
                "exists": False,
                "error": "Dataset directory does not exist."
            }

        all_items = self.discover_images()
        counts_by_class = {c: 0 for c in self.class_names}
        counts_by_folder = {}
        unreadable_files = []
        sample_resolutions = []

        for item in all_items:
            counts_by_class[item["label"]] += 1
            folder_key = item["path"].parent.name
            counts_by_folder[folder_key] = counts_by_folder.get(folder_key, 0) + 1

        # Check sample readability (up to 50 files)
        sample_check = random.sample(all_items, min(50, len(all_items))) if all_items else []
        for item in sample_check:
            try:
                with Image.open(item["path"]) as im:
                    im.verify()
                with Image.open(item["path"]) as im:
                    sample_resolutions.append(im.size)
            except Exception as e:
                unreadable_files.append({"path": item["relative_path"], "error": str(e)})

        return {
            "dataset_dir": str(self.dataset_dir),
            "exists": True,
            "total_images": len(all_items),
            "counts_by_class": counts_by_class,
            "counts_by_folder": counts_by_folder,
            "unreadable_samples_count": len(unreadable_files),
            "unreadable_samples": unreadable_files,
            "sample_resolutions": list(set(sample_resolutions))[:5]
        }

    def load_image_as_array(self, image_path: Path) -> np.ndarray:
        """
        Loads and preprocesses a single image using the shared pipeline:
        Opens -> Converts to Grayscale ('L') -> Resizes -> Normalizes -> (1, H, W)
        """
        with Image.open(image_path) as img:
            img = img.convert("L")
            img = img.resize(self.target_size, Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.float32) / config.NORMALIZATION_SCALE
            # Shape (1, H, W)
            arr = np.expand_dims(arr, axis=0)
            return arr

    def load_dataset(
        self,
        sub_path: Optional[str] = None,
        max_samples_per_class: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads images into NumPy arrays X: (N, 1, H, W) and y: (N,).
        """
        items = self.discover_images(sub_path)
        if not items:
            raise ValueError(f"No valid images found under {self.dataset_dir / (sub_path or '')}")

        if max_samples_per_class:
            class_buckets = {c: [] for c in self.class_names}
            for it in items:
                class_buckets[it["label"]].append(it)
            limited_items = []
            for c in self.class_names:
                limited_items.extend(class_buckets[c][:max_samples_per_class])
            items = limited_items

        if shuffle:
            random.seed(seed)
            random.shuffle(items)

        X_list = []
        y_list = []

        for it in items:
            try:
                arr = self.load_image_as_array(it["path"])
                X_list.append(arr)
                y_list.append(it["class_idx"])
            except Exception as e:
                # Skip corrupted file and continue
                continue

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    def get_splits(
        self,
        sub_path: Optional[str] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, List[Dict]]:
        """
        Generates reproducible train/val/test splits without duplicating or moving files.
        """
        items = self.discover_images(sub_path)
        random.seed(seed)
        random.shuffle(items)

        # Stratified splitting by class
        by_class = {c: [] for c in self.class_names}
        for it in items:
            by_class[it["label"]].append(it)

        train_set, val_set, test_set = [], [], []

        for c, class_items in by_class.items():
            n = len(class_items)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_set.extend(class_items[:n_train])
            val_set.extend(class_items[n_train:n_train + n_val])
            test_set.extend(class_items[n_train + n_val:])

        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set
        }

