"""
STATISTICAL ANALYSIS, CORRELATION MATRIX & VISUAL DASHBOARD FOR RING
----------------------------------------------------------------------------
Generates quantitative statistics and visual figures across the Test Dataset:
1. Correlation Matrix Heatmap (CNN Prob, Sobel Gradient, Black-Hat p95,
   Disparity Matrix Mean/Max, Box Count, Ground Truth).
2. 4-Panel Statistical Dashboard:
   - Confusion Matrix (Dual-Gate) & Classification Metrics
   - Scatter Plot: CNN Defect Probability (%) vs Sobel Gradient (Gate 2 Th=26.0)
   - Feature Distribution (Normal vs Defect)
   - Gate Contribution & Average Defect Boxes per Class
3. Exports figures to `results/ring_correlation_matrix.png` and
   `results/ring_statistical_dashboard.png`.
"""

import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure Unified_Inspection root is on sys.path
RING_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_ROOT = RING_ROOT.parent.parent
if str(UNIFIED_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIFIED_ROOT))
if str(RING_ROOT) not in sys.path:
    sys.path.insert(0, str(RING_ROOT))

from unified_app.adapters.ring_adapter import RingAdapter


def compute_pearson_corr(data_matrix: np.ndarray) -> np.ndarray:
    """Computes Pearson correlation matrix for columns of data_matrix."""
    return np.corrcoef(data_matrix, rowvar=False)


def main():
    test_dir = RING_ROOT / "dataset_split" / "test"
    results_dir = RING_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not test_dir.exists():
        print(f"[ERROR] Test dataset directory not found: {test_dir}")
        return

    print("=== [RING STATISTICAL REPORT] Initializing RingAdapter ===")
    adapter = RingAdapter()
    if not adapter.is_ready:
        print(f"[ERROR] RingAdapter failed to load: {adapter.load_error}")
        return

    records = []
    class_map = [("defect_ring", 1), ("normal_ring", 0)]

    for folder_name, gt_label in class_map:
        folder = test_dir / folder_name
        if not folder.exists():
            continue
        img_paths = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        )
        print(f"Processing {len(img_paths)} images from '{folder_name}'...")
        for p in img_paths:
            res = adapter.inspect(p.read_bytes(), grid_dim=512)
            if res.get("status") != "success":
                continue
            m = res["metrics"]
            records.append({
                "filename": p.name,
                "ground_truth": gt_label,
                "pred_label": 1 if res["is_defect"] else 0,
                "cnn_defect_prob": m["defect_probability"],
                "sobel_gradient": m["sobel_surface_gradient"],
                "blackhat_p95": m["blackhat_p95_disparity"],
                "matrix_mean": m["matrix_mean"],
                "matrix_max": m["matrix_max"],
                "num_boxes": m["anomaly_boxes_count"],
                "gate_info": res["gate_info"],
                "latency_ms": res["processing_ms"],
            })

    if not records:
        print("[ERROR] No test images were processed.")
        return

    # Save CSV summary
    csv_path = results_dir / "ring_statistical_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Extract arrays
    gt = np.array([r["ground_truth"] for r in records])
    pred = np.array([r["pred_label"] for r in records])
    cnn_prob = np.array([r["cnn_defect_prob"] for r in records])
    sobel = np.array([r["sobel_gradient"] for r in records])
    bh_p95 = np.array([r["blackhat_p95"] for r in records])
    mat_mean = np.array([r["matrix_mean"] for r in records])
    mat_max = np.array([r["matrix_max"] for r in records])
    num_boxes = np.array([r["num_boxes"] for r in records])
    latency = np.array([r["latency_ms"] for r in records])

    # Confusion Matrix & Classification Metrics
    tp = int(np.sum((gt == 1) & (pred == 1)))
    fn = int(np.sum((gt == 1) & (pred == 0)))
    fp = int(np.sum((gt == 0) & (pred == 1)))
    tn = int(np.sum((gt == 0) & (pred == 0)))

    acc = (tp + tn) / len(gt) * 100.0
    prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # Gate counts
    gate_counts = {}
    for r in records:
        g = r["gate_info"]
        gate_counts[g] = gate_counts.get(g, 0) + 1

    print("\n=== QUANTITATIVE SUMMARY (N = {}) ===".format(len(records)))
    print(f"Accuracy : {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1-Score: {f1:.2f}%")
    print(f"Confusion: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(f"Avg Latency: {np.mean(latency):.2f} ms (std: {np.std(latency):.2f} ms)")
    print(f"Gate Breakdown: {gate_counts}")
    print(f"Avg Defect Boxes -> Defect: {np.mean(num_boxes[gt==1]):.2f} | Normal: {np.mean(num_boxes[gt==0]):.2f}")

    # -------------------------------------------------------------------------
    # FIGURE 1: CORRELATION MATRIX HEATMAP
    # -------------------------------------------------------------------------
    feat_names = [
        "Ground Truth",
        "CNN Prob (%)",
        "Sobel Grad",
        "Black-Hat p95",
        "Matrix Mean",
        "Matrix Max",
        "Defect Boxes",
    ]
    data_mat = np.column_stack([gt, cnn_prob, sobel, bh_p95, mat_mean, mat_max, num_boxes])
    corr = compute_pearson_corr(data_mat)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation (r)", fontsize=10)

    ax.set_xticks(np.arange(len(feat_names)))
    ax.set_yticks(np.arange(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(feat_names, fontsize=10)
    ax.set_title(
        f"Matriks Korelasi Fitur Inspeksi Ring (N={len(records)})",
        fontsize=12, fontweight="bold", pad=14
    )

    for i in range(len(feat_names)):
        for j in range(len(feat_names)):
            val = corr[i, j]
            txt_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    corr_path = results_dir / "ring_correlation_matrix.png"
    plt.savefig(corr_path, dpi=180)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 2: 4-PANEL STATISTICAL DASHBOARD
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"Dashboard Statistik & Performa Modul Ring (N={len(records)} Test Images)",
        fontsize=14, fontweight="bold", y=0.98
    )

    # Panel 1: Confusion Matrix
    ax1 = axes[0, 0]
    cm = np.array([[tp, fn], [fp, tn]])
    ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Pred: Defect", "Pred: Normal"])
    ax1.set_yticklabels(["Actual: Defect", "Actual: Normal"])
    ax1.set_title(f"Confusion Matrix\nAcc={acc:.2f}% | Prec={prec:.2f}% | Rec={rec:.2f}% | F1={f1:.2f}%", fontsize=10.5, fontweight="bold")
    for i in range(2):
        for j in range(2):
            c_txt = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax1.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=c_txt, fontsize=15, fontweight="bold")

    # Panel 2: Scatter Plot CNN Prob vs Sobel Gradient
    ax2 = axes[0, 1]
    ax2.scatter(cnn_prob[gt == 1], sobel[gt == 1], c="#DC2626", alpha=0.8, label="Actual: Defect", edgecolors="k", s=55)
    ax2.scatter(cnn_prob[gt == 0], sobel[gt == 0], c="#059669", alpha=0.8, label="Actual: Normal", edgecolors="k", s=55)
    ax2.axvline(50.0, color="#2563EB", linestyle="--", linewidth=1.2, label="Gate 1 CNN (50%)")
    ax2.axhline(26.0, color="#D97706", linestyle="--", linewidth=1.2, label="Gate 2 Sobel (26.0)")
    ax2.set_xlabel("Pure CNN Defect Probability (%)")
    ax2.set_ylabel("Sobel Surface Gradient")
    ax2.set_title("Sebaran Keputusan Gate 1 (CNN) vs Gate 2 (Sobel)", fontsize=10.5, fontweight="bold")
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.grid(True, alpha=0.25)

    # Panel 3: Boxplot Comparison (Sobel & Matrix Mean)
    ax3 = axes[1, 0]
    box_data = [
        sobel[gt == 0], sobel[gt == 1],
        mat_mean[gt == 0], mat_mean[gt == 1],
    ]
    bp = ax3.boxplot(box_data, patch_artist=True, tick_labels=["Sobel\n(Normal)", "Sobel\n(Defect)", "Matrix Mean\n(Normal)", "Matrix Mean\n(Defect)"])
    colors_bp = ["#A7F3D0", "#FECACA", "#A7F3D0", "#FECACA"]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
    ax3.set_title("Distribusi Fitur Metrologi: Normal vs Defect", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("Nilai Fitur")
    ax3.grid(True, axis="y", alpha=0.25)

    # Panel 4: Gate Contribution & Mean Stats Bar Chart
    ax4 = axes[1, 1]
    labels_g = list(gate_counts.keys())
    vals_g = [gate_counts[k] for k in labels_g]
    bars = ax4.bar(range(len(labels_g)), vals_g, color=["#2563EB", "#059669", "#D97706"][:len(labels_g)], edgecolor="black")
    ax4.set_xticks(range(len(labels_g)))
    ax4.set_xticklabels(labels_g, rotation=12, ha="right", fontsize=9)
    ax4.set_ylabel("Jumlah Citra Test")
    ax4.set_title("Kontribusi Gate Keputusan pada Dataset Test", fontsize=10.5, fontweight="bold")
    for bar, v in zip(bars, vals_g):
        ax4.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v} ({v/len(records)*100:.1f}%)", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax4.set_ylim(0, max(vals_g) * 1.18)
    ax4.grid(True, axis="y", alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    dash_path = results_dir / "ring_statistical_dashboard.png"
    plt.savefig(dash_path, dpi=180)
    plt.close(fig)

    print(f"\n[SAVED] {corr_path}")
    print(f"[SAVED] {dash_path}")
    print(f"[SAVED] {csv_path}")


if __name__ == "__main__":
    main()
