"""
Dataset Audit Tool for Casting Defect Inspection.
Scans the local dataset folder, reports image counts, classes, supported formats,
checks for corrupt files, and verifies overall dataset integrity.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnn.dataset_loader import CastingDatasetLoader
import config


def run_audit():
    print("=" * 60)
    print("  CASTING PRODUCT DATASET AUDIT REPORT")
    print("=" * 60)
    print(f"Target Dataset Path : {config.DATASET_DIR}")

    loader = CastingDatasetLoader()
    result = loader.audit()

    if not result.get("exists"):
        print(f"\n[ERROR] Dataset folder not found: {result.get('dataset_dir')}")
        print("Please verify the DATASET_DIR setting or ensure the dataset is placed at ./dataset/")
        sys.exit(1)

    print(f"Total Discovered Images : {result['total_images']:,}")
    print("\n[Class Distribution]")
    for cls_name, count in result["counts_by_class"].items():
        pct = (count / result['total_images'] * 100) if result['total_images'] > 0 else 0
        print(f"  - {cls_name:<12}: {count:>5} images ({pct:.1f}%)")

    print("\n[Folder Distribution]")
    for folder_name, count in result["counts_by_folder"].items():
        print(f"  - {folder_name:<16}: {count:>5} images")

    print("\n[Sample Resolutions Detected]")
    for res in result["sample_resolutions"]:
        print(f"  - Width: {res[0]}px, Height: {res[1]}px")

    print(f"\n[Integrity Verification]")
    print(f"  - Unreadable / Corrupt Samples: {result['unreadable_samples_count']}")
    if result["unreadable_samples_count"] > 0:
        print("  Warning: Detected corrupted samples:")
        for bad in result["unreadable_samples"]:
            print(f"    * {bad['path']}: {bad['error']}")
    else:
        print("  - All checked samples verified readable.")

    print("\n" + "=" * 60)
    print("  AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_audit()

