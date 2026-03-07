import numpy as np


# --------------------------------------------------
# Sigmoid Activation
# --------------------------------------------------

class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, X):
        # ✅ FIX: numerically stable sigmoid
        # avoids overflow for large negative values
        pos_mask = X >= 0
        result = np.empty_like(X, dtype=float)
        result[pos_mask]  = 1.0 / (1.0 + np.exp(-X[pos_mask]))
        exp_x = np.exp(X[~pos_mask])
        result[~pos_mask] = exp_x / (1.0 + exp_x)
        self.output = result
        return self.output

    def backward(self, dA):
        # derivative of sigmoid: s * (1 - s)
        return dA * (self.output * (1 - self.output))


# --------------------------------------------------
# Tanh Activation
# --------------------------------------------------

class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, X):
        self.output = np.tanh(X)
        return self.output

    def backward(self, dA):
        # derivative of tanh: 1 - tanh^2
        return dA * (1 - self.output ** 2)


# --------------------------------------------------
# ReLU Activation
# --------------------------------------------------

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, X):
        self.input = X
        return np.maximum(0.0, X)

    def backward(self, dA):
        # ✅ FIX: cleaner derivative using boolean mask
        return dA * (self.input > 0).astype(float)


# --------------------------------------------------
# Softmax Activation (used only for inference/output, not in backprop)
# Softmax + loss gradient is handled inside objective_functions.py
# --------------------------------------------------

class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, X):
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dA):
        # gradient is handled inside objective_functions.py (combined with loss)
        return dA