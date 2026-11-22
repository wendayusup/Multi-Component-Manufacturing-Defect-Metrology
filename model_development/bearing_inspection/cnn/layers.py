"""
Custom Pure-NumPy Neural Network Layers.
Implements Conv2D, ReLU, MaxPool2D, Flatten, and Dense layers without external deep learning frameworks.
Vectorized using im2col / col2im for high performance in NumPy.
"""
import numpy as np


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """
    Transforms image tensor into 2D matrix columns for fast vectorized convolution.
    Input shape: (N, C, H, W)
    """
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    N, C, H, W = x.shape
    out_height = int((H + 2 * p - field_height) / stride + 1)
    out_width = int((W + 2 * p - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """
    Transforms column gradients back into input gradient image tensor.
    """
    N, C, H, W = x_shape
    p = padding
    H_padded, W_padded = H + 2 * p, W + 2 * p
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    out_height = int((H + 2 * p - field_height) / stride + 1)
    out_width = int((W + 2 * p - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if p == 0:
        return x_padded
    return x_padded[:, :, p:-p, p:-p]


class Layer:
    """Base class for neural network layers."""
    def forward(self, x, training=False):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def get_params(self):
        return []

    def get_grads(self):
        return []

    def set_params(self, params):
        pass


class Conv2D(Layer):
    """
    2D Convolutional Layer.
    Weights shape: (num_filters, in_channels, kernel_size, kernel_size)
    Biases shape: (num_filters, 1)
    """
    def __init__(self, in_channels, num_filters, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He (Kaiming) normal initialization
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.W = np.random.randn(num_filters, in_channels, kernel_size, kernel_size) * std
        self.b = np.zeros((num_filters, 1))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.last_x = None
        self.x_cols = None

    def forward(self, x, training=False):
        self.last_x = x
        N, C, H, W = x.shape
        out_h = int((H + 2 * self.padding - self.kernel_size) / self.stride + 1)
        out_w = int((W + 2 * self.padding - self.kernel_size) / self.stride + 1)

        self.x_cols = im2col_indices(x, self.kernel_size, self.kernel_size,
                                     padding=self.padding, stride=self.stride)
        w_row = self.W.reshape(self.num_filters, -1)

        out = w_row @ self.x_cols + self.b
        out = out.reshape(self.num_filters, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)  # (N, num_filters, out_h, out_w)
        return out

    def backward(self, grad_output):
        # grad_output: (N, num_filters, out_h, out_w)
        N = grad_output.shape[0]
        grad_flat = grad_output.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)

        w_row = self.W.reshape(self.num_filters, -1)
        self.dW = (grad_flat @ self.x_cols.T).reshape(self.W.shape)
        self.db = np.sum(grad_flat, axis=1, keepdims=True)

        dcols = w_row.T @ grad_flat
        dx = col2im_indices(dcols, self.last_x.shape, self.kernel_size, self.kernel_size,
                            padding=self.padding, stride=self.stride)
        return dx

    def get_params(self):
        return [self.W, self.b]

    def get_grads(self):
        return [self.dW, self.db]

    def set_params(self, params):
        self.W = params[0].copy()
        self.b = params[1].copy()


class ReLU(Layer):
    """Rectified Linear Unit activation layer."""
    def __init__(self):
        self.last_x = None

    def forward(self, x, training=False):
        self.last_x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.last_x > 0)


class MaxPool2D(Layer):
    """
    Max Pooling Layer with stride equal to pool_size.
    Input shape: (N, C, H, W)
    """
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.last_x = None
        self.x_reshaped = None
        self.out = None

    def forward(self, x, training=False):
        self.last_x = x
        N, C, H, W = x.shape
        ph, pw = self.pool_size, self.pool_size
        sh, sw = self.stride, self.stride

        out_h = int((H - ph) / sh + 1)
        out_w = int((W - pw) / sw + 1)

        # Efficient reshaped max pooling for non-overlapping windows (stride == pool_size)
        if sh == ph and sw == pw and H % ph == 0 and W % pw == 0:
            x_reshaped = x.reshape(N, C, out_h, ph, out_w, pw)
            self.x_reshaped = x_reshaped
            out = x_reshaped.max(axis=(3, 5))
            self.out = out
            return out
        else:
            # Fallback for arbitrary pool / stride using col conversion
            x_cols = im2col_indices(x, ph, pw, padding=0, stride=sh)
            x_cols = x_cols.reshape(C, ph * pw, -1)
            out_idx = np.argmax(x_cols, axis=1)
            out = np.max(x_cols, axis=1)
            out = out.reshape(C, out_h, out_w, N).transpose(3, 0, 1, 2)
            self.x_cols = x_cols
            self.out_idx = out_idx
            return out

    def backward(self, grad_output):
        N, C, H, W = self.last_x.shape
        ph, pw = self.pool_size, self.pool_size
        sh, sw = self.stride, self.stride
        out_h = int((H - ph) / sh + 1)
        out_w = int((W - pw) / sw + 1)

        if sh == ph and sw == pw and H % ph == 0 and W % pw == 0:
            dx = np.zeros_like(self.last_x)
            dx_reshaped = dx.reshape(N, C, out_h, ph, out_w, pw)
            out_expanded = self.out[:, :, :, np.newaxis, :, np.newaxis]
            grad_expanded = grad_output[:, :, :, np.newaxis, :, np.newaxis]
            mask = (self.x_reshaped == out_expanded)
            # Normalize in case of duplicate max values in window
            mask = mask / np.maximum(mask.sum(axis=(3, 5), keepdims=True), 1e-12)
            dx_reshaped += mask * grad_expanded
            return dx
        else:
            dcols = np.zeros_like(self.x_cols)
            idx = np.expand_dims(self.out_idx, axis=1)
            np.put_along_axis(dcols, idx, np.expand_dims(grad_output.transpose(1, 2, 3, 0).reshape(C, -1), axis=1), axis=1)
            dcols = dcols.reshape(C * ph * pw, -1)
            dx = col2im_indices(dcols, self.last_x.shape, ph, pw, padding=0, stride=sh)
            return dx


class Flatten(Layer):
    """Flattens (N, C, H, W) to (N, C * H * W)."""
    def __init__(self):
        self.orig_shape = None

    def forward(self, x, training=False):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.orig_shape)


class Dense(Layer):
    """
    Fully Connected (Dense) Layer.
    Weights shape: (in_features, out_features)
    Biases shape: (1, out_features)
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # He (Kaiming) initialization
        std = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * std
        self.b = np.zeros((1, out_features))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.last_x = None

    def forward(self, x, training=False):
        self.last_x = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        self.dW = self.last_x.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T

    def get_params(self):
        return [self.W, self.b]

    def get_grads(self):
        return [self.dW, self.db]

    def set_params(self, params):
        self.W = params[0].copy()
        self.b = params[1].copy()

