import numpy as np


# ======================================================================
# SGD (vanilla)
# ======================================================================

class SGD:
    """Vanilla stochastic gradient descent."""

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


# ======================================================================
# Momentum SGD
# ======================================================================

class Momentum:
    """SGD with classical momentum."""

    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr   = learning_rate
        self.beta = beta
        self._vW  = None
        self._vb  = None

    def _init_state(self, layers):
        self._vW = [np.zeros_like(l.W) for l in layers]
        self._vb = [np.zeros_like(l.b) for l in layers]

    def step(self, layers):
        if self._vW is None:
            self._init_state(layers)

        for i, layer in enumerate(layers):
            # ✅ FIX: include lr in velocity accumulation (correct formula)
            self._vW[i] = self.beta * self._vW[i] + self.lr * layer.grad_W
            self._vb[i] = self.beta * self._vb[i] + self.lr * layer.grad_b
            layer.W -= self._vW[i]
            layer.b -= self._vb[i]


# ======================================================================
# NAG (Nesterov Accelerated Gradient)
# ======================================================================

class NAG:
    """Nesterov Accelerated Gradient."""

    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr   = learning_rate
        self.beta = beta
        self._vW  = None
        self._vb  = None

    def _init_state(self, layers):
        self._vW = [np.zeros_like(l.W) for l in layers]
        self._vb = [np.zeros_like(l.b) for l in layers]

    def step(self, layers):
        if self._vW is None:
            self._init_state(layers)

        for i, layer in enumerate(layers):
            # ✅ FIX: correct Nesterov update using previous velocity
            vW_prev = self._vW[i].copy()
            vb_prev = self._vb[i].copy()

            self._vW[i] = self.beta * self._vW[i] + self.lr * layer.grad_W
            self._vb[i] = self.beta * self._vb[i] + self.lr * layer.grad_b

            # Nesterov correction: look-ahead using updated velocity
            layer.W -= (1 + self.beta) * self._vW[i] - self.beta * vW_prev
            layer.b -= (1 + self.beta) * self._vb[i] - self.beta * vb_prev


# ======================================================================
# RMSProp
# ======================================================================

class RMSProp:
    """RMSProp — adaptive per-parameter learning rates."""

    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.lr      = learning_rate
        self.beta    = beta
        self.epsilon = epsilon
        self._sW     = None
        self._sb     = None

    def _init_state(self, layers):
        self._sW = [np.zeros_like(l.W) for l in layers]
        self._sb = [np.zeros_like(l.b) for l in layers]

    def step(self, layers):
        if self._sW is None:
            self._init_state(layers)

        for i, layer in enumerate(layers):
            self._sW[i] = self.beta * self._sW[i] + (1 - self.beta) * layer.grad_W ** 2
            self._sb[i] = self.beta * self._sb[i] + (1 - self.beta) * layer.grad_b ** 2
            layer.W -= self.lr * layer.grad_W / (np.sqrt(self._sW[i]) + self.epsilon)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self._sb[i]) + self.epsilon)


# ======================================================================
# Adam
# ======================================================================

class Adam:
    """Adam optimizer (Kingma & Ba, 2015)."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr      = learning_rate
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self._t      = 0
        self._mW     = None
        self._mb     = None
        self._vW     = None
        self._vb     = None

    def _init_state(self, layers):
        self._mW = [np.zeros_like(l.W) for l in layers]
        self._mb = [np.zeros_like(l.b) for l in layers]
        self._vW = [np.zeros_like(l.W) for l in layers]
        self._vb = [np.zeros_like(l.b) for l in layers]

    def step(self, layers):
        if self._mW is None:
            self._init_state(layers)

        self._t += 1

        # ✅ FIX: compute bias-corrected lr once upfront (cleaner + correct)
        lr_t = self.lr * (np.sqrt(1 - self.beta2 ** self._t) / (1 - self.beta1 ** self._t))

        for i, layer in enumerate(layers):
            # First moment
            self._mW[i] = self.beta1 * self._mW[i] + (1 - self.beta1) * layer.grad_W
            self._mb[i] = self.beta1 * self._mb[i] + (1 - self.beta1) * layer.grad_b
            # Second moment
            self._vW[i] = self.beta2 * self._vW[i] + (1 - self.beta2) * layer.grad_W ** 2
            self._vb[i] = self.beta2 * self._vb[i] + (1 - self.beta2) * layer.grad_b ** 2
            # Update
            layer.W -= lr_t * self._mW[i] / (np.sqrt(self._vW[i]) + self.epsilon)
            layer.b -= lr_t * self._mb[i] / (np.sqrt(self._vb[i]) + self.epsilon)


# ======================================================================
# Nadam
# ======================================================================

class Nadam:
    """Nadam — Adam with Nesterov momentum (Dozat, 2016)."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr      = learning_rate
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self._t      = 0
        self._mW     = None
        self._mb     = None
        self._vW     = None
        self._vb     = None

    def _init_state(self, layers):
        self._mW = [np.zeros_like(l.W) for l in layers]
        self._mb = [np.zeros_like(l.b) for l in layers]
        self._vW = [np.zeros_like(l.W) for l in layers]
        self._vb = [np.zeros_like(l.b) for l in layers]

    def step(self, layers):
        if self._mW is None:
            self._init_state(layers)

        self._t += 1

        for i, layer in enumerate(layers):
            # First moment
            self._mW[i] = self.beta1 * self._mW[i] + (1 - self.beta1) * layer.grad_W
            self._mb[i] = self.beta1 * self._mb[i] + (1 - self.beta1) * layer.grad_b
            # Second moment
            self._vW[i] = self.beta2 * self._vW[i] + (1 - self.beta2) * layer.grad_W ** 2
            self._vb[i] = self.beta2 * self._vb[i] + (1 - self.beta2) * layer.grad_b ** 2
            # Bias-corrected estimates
            mW_hat = self._mW[i] / (1 - self.beta1 ** self._t)
            mb_hat = self._mb[i] / (1 - self.beta1 ** self._t)
            vW_hat = self._vW[i] / (1 - self.beta2 ** self._t)
            vb_hat = self._vb[i] / (1 - self.beta2 ** self._t)
            # ✅ Nesterov look-ahead: blend next step momentum
            mW_nesterov = self.beta1 * mW_hat + (1 - self.beta1) * layer.grad_W / (1 - self.beta1 ** self._t)
            mb_nesterov = self.beta1 * mb_hat + (1 - self.beta1) * layer.grad_b / (1 - self.beta1 ** self._t)
            layer.W -= self.lr * mW_nesterov / (np.sqrt(vW_hat) + self.epsilon)
            layer.b -= self.lr * mb_nesterov / (np.sqrt(vb_hat) + self.epsilon)