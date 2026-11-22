"""
Optimizers for Training the Custom NumPy CNN.
Provides SGD (with momentum and weight decay) and Adam optimizers.
"""
import numpy as np


class SGD:
    """Stochastic Gradient Descent with optional momentum and weight decay."""
    def __init__(self, layers, lr=0.01, momentum=0.9, weight_decay=1e-4):
        self.layers = layers
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = []

        for layer in layers:
            layer_vel = [np.zeros_like(p) for p in layer.get_params()]
            self.velocities.append(layer_vel)

    def step(self):
        for layer_idx, layer in enumerate(self.layers):
            params = layer.get_params()
            grads = layer.get_grads()
            velocities = self.velocities[layer_idx]

            for i in range(len(params)):
                grad = grads[i]
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * params[i]

                if self.momentum > 0:
                    velocities[i] = self.momentum * velocities[i] - self.lr * grad
                    params[i] += velocities[i]
                else:
                    params[i] -= self.lr * grad


class Adam:
    """Adam Optimizer (Adaptive Moment Estimation)."""
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = []
        self.v = []
        for layer in layers:
            self.m.append([np.zeros_like(p) for p in layer.get_params()])
            self.v.append([np.zeros_like(p) for p in layer.get_params()])

    def step(self):
        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)

        for layer_idx, layer in enumerate(self.layers):
            params = layer.get_params()
            grads = layer.get_grads()

            for i in range(len(params)):
                g = grads[i]
                if self.weight_decay > 0:
                    g = g + self.weight_decay * params[i]

                # Update biased 1st and 2nd moment estimate
                self.m[layer_idx][i] = self.beta1 * self.m[layer_idx][i] + (1.0 - self.beta1) * g
                self.v[layer_idx][i] = self.beta2 * self.v[layer_idx][i] + (1.0 - self.beta2) * (g ** 2)

                # Parameter update
                update = lr_t * self.m[layer_idx][i] / (np.sqrt(self.v[layer_idx][i]) + self.eps)
                params[i] -= update

