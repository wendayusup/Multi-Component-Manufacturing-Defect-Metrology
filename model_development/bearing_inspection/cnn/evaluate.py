"""
Evaluation Module for Custom Casting CNN.
Computes Confusion Matrix, Precision, Recall, F1 Score per class, and Macro F1.
"""
from typing import Dict, List, Tuple
import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict:
    """
    Computes classification performance metrics:
    - Confusion Matrix
    - Accuracy
    - Per-class Precision, Recall, F1
    - Macro Average F1
    """
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    accuracy = float(np.sum(np.diag(cm)) / max(len(y_true), 1))

    per_class = {}
    f1_scores = []

    for i, c_name in enumerate(class_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        f1_scores.append(f1)
        per_class[c_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": int(np.sum(cm[i, :]))
        }

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def format_evaluation_report(metrics: Dict, class_names: List[str]) -> str:
    """Formats metrics as a human-readable table."""
    lines = []
    lines.append("-" * 65)
    lines.append(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    lines.append("-" * 65)

    for c_name in class_names:
        stats = metrics["per_class"][c_name]
        lines.append(
            f"{c_name:<14} | {stats['precision']:<10.4f} | {stats['recall']:<10.4f} | "
            f"{stats['f1_score']:<10.4f} | {stats['support']:<8}"
        )

    lines.append("-" * 65)
    lines.append(f"Overall Accuracy: {metrics['accuracy'] * 100:.2f}%")
    lines.append(f"Macro F1 Score  : {metrics['macro_f1']:.4f}")
    lines.append("\nConfusion Matrix (Rows: True, Cols: Predicted):")
    lines.append(f"           {'  '.join([f'{c[:5]:>5}' for c in class_names])}")
    for idx, row in enumerate(metrics["confusion_matrix"]):
        lines.append(f"  {class_names[idx]:<8} {'  '.join([f'{val:>5}' for val in row])}")
    lines.append("-" * 65)
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config
    from cnn.model import build_casting_cnn
    from cnn.dataset_loader import CastingDatasetLoader

    parser = argparse.ArgumentParser(description="Evaluate Custom Casting CNN on test dataset")
    parser.add_argument("--weights", type=str, default=str(config.DEFAULT_WEIGHTS_FILE), help="Weights path")
    parser.add_argument("--samples", type=int, default=300, help="Number of test samples")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] Weights file not found: {weights_path}")
        sys.exit(1)

    print("=" * 65)
    print("  OFFLINE EVALUATION: CUSTOM CASTING CNN")
    print("=" * 65)
    print(f"Loading weights from: {weights_path}")

    model = build_casting_cnn(input_shape=(1, 64, 64), num_classes=len(config.CLASS_NAMES))
    model.load_weights(weights_path)

    loader = CastingDatasetLoader()
    test_subpath = "casting_data/casting_data/test"
    if not (config.DATASET_DIR / test_subpath).exists():
        test_subpath = None

    print(f"Loading {args.samples} test samples from dataset...")
    X_test, y_test = loader.load_dataset(
        sub_path=test_subpath,
        max_samples_per_class=args.samples // 2,
        shuffle=True,
        seed=100
    )

    logits = model.forward(X_test, training=False)
    preds = np.argmax(logits, axis=1)

    metrics = compute_metrics(y_test, preds, config.CLASS_NAMES)
    print("\n" + format_evaluation_report(metrics, config.CLASS_NAMES))
    print("Evaluation completed successfully.")

