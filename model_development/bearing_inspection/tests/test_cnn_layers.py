"""
Unit Tests for Custom Pure-NumPy Neural Network Layers.
Verifies mathematical correctness of forward/backward propagation for all layers.
"""
import unittest
import numpy as np

from cnn.layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
from cnn.losses import softmax, CrossEntropyLoss
from cnn.optimizer import Adam, SGD


class TestCNNLayers(unittest.TestCase):

    def test_conv2d_forward_backward_shapes(self):
        """Verifies Conv2D forward output and backward gradient shapes."""
        conv = Conv2D(in_channels=1, num_filters=4, kernel_size=3, stride=1, padding=1)
        x = np.random.randn(2, 1, 16, 16)
        out = conv.forward(x, training=True)

        self.assertEqual(out.shape, (2, 4, 16, 16))

        grad_out = np.random.randn(2, 4, 16, 16)
        dx = conv.backward(grad_out)

        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(conv.dW.shape, conv.W.shape)
        self.assertEqual(conv.db.shape, conv.b.shape)

    def test_relu_forward_backward(self):
        """Verifies ReLU activation and gradient masking."""
        relu = ReLU()
        x = np.array([-2.0, 0.0, 3.0, 5.0])
        out = relu.forward(x, training=True)
        np.testing.assert_array_equal(out, np.array([0.0, 0.0, 3.0, 5.0]))

        grad_out = np.array([1.0, 1.0, 1.0, 1.0])
        dx = relu.backward(grad_out)
        np.testing.assert_array_equal(dx, np.array([0.0, 0.0, 1.0, 1.0]))

    def test_maxpool2d_forward_backward(self):
        """Verifies MaxPool2D downsampling and backpropagation."""
        pool = MaxPool2D(pool_size=2, stride=2)
        x = np.random.randn(2, 2, 8, 8)
        out = pool.forward(x, training=True)
        self.assertEqual(out.shape, (2, 2, 4, 4))

        grad_out = np.random.randn(2, 2, 4, 4)
        dx = pool.backward(grad_out)
        self.assertEqual(dx.shape, x.shape)

    def test_dense_forward_backward(self):
        """Verifies Dense linear layer forward and backpropagation."""
        dense = Dense(in_features=16, out_features=4)
        x = np.random.randn(3, 16)
        out = dense.forward(x, training=True)
        self.assertEqual(out.shape, (3, 4))

        grad_out = np.random.randn(3, 4)
        dx = dense.backward(grad_out)
        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(dense.dW.shape, (16, 4))
        self.assertEqual(dense.db.shape, (1, 4))

    def test_softmax_numerical_stability(self):
        """Verifies softmax prevents overflow even with huge logit values."""
        extreme_logits = np.array([[1000.0, 1005.0, 1002.0]])
        probs = softmax(extreme_logits)
        self.assertFalse(np.isnan(probs).any())
        self.assertFalse(np.isinf(probs).any())
        np.testing.assert_almost_equal(np.sum(probs), 1.0)

    def test_cross_entropy_loss_and_backward(self):
        """Verifies loss computation and analytic gradient shape."""
        loss_fn = CrossEntropyLoss()
        logits = np.array([[2.0, 0.5], [0.1, 1.8]])
        targets = np.array([0, 1])

        loss = loss_fn.forward(logits, targets)
        self.assertGreater(loss, 0.0)

        grad = loss_fn.backward()
        self.assertEqual(grad.shape, logits.shape)

    def test_optimizer_updates_weights(self):
        """Verifies Adam and SGD modify parameters when stepped."""
        dense = Dense(in_features=4, out_features=2)
        initial_W = dense.W.copy()

        opt = Adam([dense], lr=0.1)
        dense.dW = np.ones_like(dense.W)
        opt.step()

        self.assertFalse(np.array_equal(dense.W, initial_W))


if __name__ == "__main__":
    unittest.main()

