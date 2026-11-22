"""
Loss and Activation Functions for Custom CNN.
Includes numerically stable Softmax and Categorical Cross-Entropy Loss.
"""
import numpy as np


def softmax(logits):
    """
    Numerically stable softmax activation over the last dimension.
    Subtracts max per sample to prevent numerical overflow in exp.
    """
    logits = np.asarray(logits, dtype=np.float64)
    # Subtract max along last axis for numerical stability
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return probs


class CrossEntropyLoss:
    """
    Computes Categorical Cross-Entropy loss from unnormalized logits.
    Combines Softmax + Cross-Entropy for optimal numerical stability in backward pass.
    """
    def __init__(self, eps=1e-12):
        self.eps = eps
        self.last_probs = None
        self.last_targets = None

    def forward(self, logits, targets):
        """
        Forward pass.
        logits: (N, num_classes)
        targets: 1D array of class indices (N,) or one-hot (N, num_classes)
        """
        N = logits.shape[0]
        probs = softmax(logits)
        self.last_probs = probs

        if targets.ndim == 1:
            self.last_targets = targets
            correct_confidences = probs[np.arange(N), targets]
        else:
            self.last_targets = np.argmax(targets, axis=1)
            correct_confidences = np.sum(probs * targets, axis=1)

        # Negative log-likelihood
        loss = -np.mean(np.log(np.clip(correct_confidences, self.eps, 1.0)))
        return float(loss)

    def backward(self):
        """
        Backward pass returning gradient of loss with respect to logits:
        d(loss) / d(logits) = (probs - targets_one_hot) / N
        """
        probs = self.last_probs.copy()
        N = probs.shape[0]
        probs[np.arange(N), self.last_targets] -= 1.0
        return probs / N

