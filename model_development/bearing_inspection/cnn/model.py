"""
Sequential Model Architecture and Casting CNN Factory.
Provides layer chaining, forward/backward execution, weight serialization, and prediction.
"""
import os
from pathlib import Path
import numpy as np

from cnn.layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
from cnn.losses import softmax


class Sequential:
    """Sequential Neural Network container."""
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x, training=False):
        out = x
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict_proba(self, x):
        """Runs forward inference and returns softmax probabilities (N, num_classes)."""
        logits = self.forward(x, training=False)
        return softmax(logits)

    def predict(self, x):
        """Returns predicted class indices (N,)."""
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)

    def get_trainable_layers(self):
        return [l for l in self.layers if len(l.get_params()) > 0]

    def save_weights(self, filepath):
        """Saves all layer parameters to a compressed .npz archive."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        params_dict = {}
        param_idx = 0
        for l_idx, layer in enumerate(self.layers):
            params = layer.get_params()
            for p_sub_idx, param in enumerate(params):
                key = f"layer_{l_idx}_param_{p_sub_idx}"
                params_dict[key] = param
                param_idx += 1

        np.savez_compressed(filepath, **params_dict)

    def load_weights(self, filepath):
        """Loads layer parameters from a .npz archive."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model weights file not found: {filepath}")

        with np.load(filepath) as data:
            for l_idx, layer in enumerate(self.layers):
                layer_params = layer.get_params()
                if not layer_params:
                    continue
                new_params = []
                for p_sub_idx in range(len(layer_params)):
                    key = f"layer_{l_idx}_param_{p_sub_idx}"
                    if key not in data:
                        raise KeyError(f"Missing parameter '{key}' in weights file: {filepath}")
                    saved_arr = data[key]
                    expected_shape = layer_params[p_sub_idx].shape
                    if saved_arr.shape != expected_shape:
                        raise ValueError(
                            f"Shape mismatch for {key}: expected {expected_shape}, got {saved_arr.shape}"
                        )
                    new_params.append(saved_arr)
                layer.set_params(new_params)


def build_casting_cnn(input_shape=(1, 64, 64), num_classes=2):
    """
    Constructs the custom Casting Defect Inspection CNN baseline:
    1. Conv2D (1 -> 8 filters, 3x3 kernel, padding=1) + ReLU
    2. MaxPool2D (2x2) -> out (8, 32, 32)
    3. Conv2D (8 -> 16 filters, 3x3 kernel, padding=1) + ReLU
    4. MaxPool2D (2x2) -> out (16, 16, 16)
    5. Flatten -> 16 * 16 * 16 = 4096
    6. Dense (4096 -> 64) + ReLU
    7. Dense (64 -> num_classes)
    """
    in_channels, in_h, in_w = input_shape
    assert in_h == 64 and in_w == 64, f"Standard casting baseline expects 64x64, received: {input_shape}"

    model = Sequential([
        Conv2D(in_channels=in_channels, num_filters=8, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),
        Conv2D(in_channels=8, num_filters=16, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),
        Flatten(),
        Dense(in_features=16 * 16 * 16, out_features=64),
        ReLU(),
        Dense(in_features=64, out_features=num_classes)
    ])
    return model

