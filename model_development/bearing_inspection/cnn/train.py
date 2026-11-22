"""
Offline Training Script for Custom Pure-NumPy Casting CNN.
Trains baseline convolutional neural network on local casting dataset,
evaluates on test split, logs metrics, and exports weights with cryptographic checksum.
"""
import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cnn.model import build_casting_cnn
from cnn.losses import CrossEntropyLoss
from cnn.optimizer import Adam, SGD
from cnn.dataset_loader import CastingDatasetLoader
from cnn.evaluate import compute_metrics, format_evaluation_report
import config


def compute_file_sha256(filepath: Path) -> str:
    """Calculates SHA-256 hash of a file for integrity tracking."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def train(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.001,
    max_train_samples: int = 1000,
    max_test_samples: int = 300,
    optimizer_type: str = "adam",
    weights_path: Path = config.DEFAULT_WEIGHTS_FILE,
    meta_path: Path = config.DEFAULT_META_FILE,
    seed: int = 42
):
    np.random.seed(seed)
    print("=" * 65)
    print("  STARTING OFFLINE TRAINING: CUSTOM NUMPY CNN")
    print("=" * 65)
    print(f"Model Version     : {config.MODEL_VERSION}")
    print(f"Input Shape       : (1, 64, 64)")
    print(f"Classes           : {config.CLASS_NAMES}")
    print(f"Epochs            : {epochs}")
    print(f"Batch Size        : {batch_size}")
    print(f"Learning Rate     : {lr}")
    print(f"Max Train Samples : {max_train_samples}")
    print(f"Optimizer         : {optimizer_type.upper()}")

    loader = CastingDatasetLoader()

    # Load dataset
    print("\n[1/4] Loading and preprocessing dataset...")
    # Prioritize loading casting_data if available
    train_subpath = "casting_data/casting_data/train"
    test_subpath = "casting_data/casting_data/test"

    if not (config.DATASET_DIR / train_subpath).exists():
        train_subpath = None
        test_subpath = None

    if train_subpath:
        X_train, y_train = loader.load_dataset(
            sub_path=train_subpath,
            max_samples_per_class=max_train_samples // 2,
            shuffle=True,
            seed=seed
        )
        X_test, y_test = loader.load_dataset(
            sub_path=test_subpath,
            max_samples_per_class=max_test_samples // 2,
            shuffle=True,
            seed=seed
        )
    else:
        # Load from general dataset and perform split
        splits = loader.get_splits(seed=seed)
        # Load array data
        X_train_list, y_train_list = [], []
        for it in splits["train"][:max_train_samples]:
            X_train_list.append(loader.load_image_as_array(it["path"]))
            y_train_list.append(it["class_idx"])
        X_train = np.stack(X_train_list, axis=0)
        y_train = np.array(y_train_list, dtype=np.int64)

        X_test_list, y_test_list = [], []
        for it in splits["test"][:max_test_samples]:
            X_test_list.append(loader.load_image_as_array(it["path"]))
            y_test_list.append(it["class_idx"])
        X_test = np.stack(X_test_list, axis=0)
        y_test = np.array(y_test_list, dtype=np.int64)

    print(f"Training set:   {X_train.shape[0]} samples, shape: {X_train.shape}")
    print(f"Evaluation set: {X_test.shape[0]} samples, shape: {X_test.shape}")

    # Build model
    print("\n[2/4] Initializing Custom Pure-NumPy CNN Architecture...")
    model = build_casting_cnn(input_shape=(1, 64, 64), num_classes=len(config.CLASS_NAMES))
    loss_fn = CrossEntropyLoss()

    trainable_layers = model.get_trainable_layers()
    if optimizer_type.lower() == "adam":
        optimizer = Adam(trainable_layers, lr=lr)
    else:
        optimizer = SGD(trainable_layers, lr=lr, momentum=0.9)

    # Training loop
    print("\n[3/4] Executing Training Loop...")
    start_time = time.time()
    num_train = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        # Shuffle training set
        indices = np.random.permutation(num_train)
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0.0
        correct = 0
        batches = int(np.ceil(num_train / batch_size))

        for b in range(batches):
            b_start = b * batch_size
            b_end = min(b_start + batch_size, num_train)

            xb = X_train[b_start:b_end]
            yb = y_train[b_start:b_end]

            # Forward
            logits = model.forward(xb, training=True)
            loss = loss_fn.forward(logits, yb)
            epoch_loss += loss * (b_end - b_start)

            preds = np.argmax(logits, axis=1)
            correct += np.sum(preds == yb)

            # Backward
            loss_grad = loss_fn.backward()
            model.backward(loss_grad)

            # Optimizer Step
            optimizer.step()

        epoch_loss /= num_train
        epoch_acc = (correct / num_train) * 100.0
        epoch_time = time.time() - epoch_start
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}% - Time: {epoch_time:.2f}s")

        # Learning rate decay for sharper convergence
        if epoch in [4, 6]:
            optimizer.lr *= 0.5
            print(f"    -> [LR Scheduler] Adjusted learning rate to: {optimizer.lr:.6f}")

    total_training_time = time.time() - start_time
    print(f"\nTotal Training Duration: {total_training_time:.2f}s")

    # Evaluation
    print("\n[4/4] Evaluating on Test Split...")
    test_logits = model.forward(X_test, training=False)
    test_preds = np.argmax(test_logits, axis=1)

    metrics = compute_metrics(y_test, test_preds, config.CLASS_NAMES)
    print("\n" + format_evaluation_report(metrics, config.CLASS_NAMES))

    # Save weights and metadata
    weights_path = Path(weights_path)
    meta_path = Path(meta_path)
    model.save_weights(weights_path)
    weights_hash = compute_file_sha256(weights_path)

    metadata = {
        "model_version": config.MODEL_VERSION,
        "module": "casting_defect",
        "component_type": "casting",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_shape": [1, 64, 64],
        "class_names": config.CLASS_NAMES,
        "weights_file": weights_path.name,
        "weights_sha256": weights_hash,
        "training_params": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": optimizer_type,
            "train_samples": int(num_train),
            "test_samples": int(X_test.shape[0]),
            "seed": seed
        },
        "evaluation_metrics": metrics
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel weights saved to : {weights_path}")
    print(f"Model metadata saved to: {meta_path}")
    print(f"SHA-256 Checksum       : {weights_hash}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pure-NumPy CNN on casting dataset")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--max-train", type=int, default=1800, help="Maximum training samples")
    parser.add_argument("--max-test", type=int, default=400, help="Maximum test samples")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--weights", type=str, default=str(config.DEFAULT_WEIGHTS_FILE))
    parser.add_argument("--meta", type=str, default=str(config.DEFAULT_META_FILE))

    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=args.max_train,
        max_test_samples=args.max_test,
        optimizer_type=args.optimizer,
        weights_path=Path(args.weights),
        meta_path=Path(args.meta)
    )

