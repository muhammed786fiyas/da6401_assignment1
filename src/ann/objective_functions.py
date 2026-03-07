import numpy as np


# --------------------------------------------------
# Helper: numerically stable softmax
# --------------------------------------------------

def _softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# --------------------------------------------------
# Helper: convert int labels to one-hot if needed
# --------------------------------------------------

def _to_one_hot(y, num_classes):
    if y.ndim == 2:
        return y  # already one-hot
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return one_hot


# --------------------------------------------------
# Mean Squared Error Loss
# --------------------------------------------------

class MeanSquaredError:

    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        """
        y_pred : raw logits  (batch_size, num_classes)
        y_true : int labels  (batch_size,) OR one-hot (batch_size, num_classes)
        """
        probs = _softmax(y_pred)                       # ✅ apply softmax to logits
        y_oh  = _to_one_hot(y_true, y_pred.shape[1])  # ✅ handle int labels

        loss = np.mean(np.sum((probs - y_oh) ** 2, axis=1))
        return float(loss)

    def backward(self, y_pred, y_true):                # ✅ accepts arguments
        """
        Proper gradient of MSE loss through softmax w.r.t logits
        """
        probs      = _softmax(y_pred)
        y_oh       = _to_one_hot(y_true, y_pred.shape[1])
        batch_size = y_pred.shape[0]

        dl_dp = 2.0 * (probs - y_oh)
        dot   = np.sum(dl_dp * probs, axis=1, keepdims=True)
        grad  = probs * (dl_dp - dot) / batch_size    # ✅ gradient through softmax
        return grad


# --------------------------------------------------
# Cross Entropy Loss
# --------------------------------------------------

class CrossEntropy:

    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        """
        y_pred : raw logits  (batch_size, num_classes)
        y_true : int labels  (batch_size,) OR one-hot (batch_size, num_classes)
        """
        probs = _softmax(y_pred)                       # ✅ apply softmax to logits
        y_oh  = _to_one_hot(y_true, y_pred.shape[1])  # ✅ handle int labels

        probs_clipped = np.clip(probs, 1e-12, 1.0)
        loss = -np.sum(y_oh * np.log(probs_clipped)) / y_pred.shape[0]
        return float(loss)

    def backward(self, y_pred, y_true):                # ✅ accepts arguments
        """
        Combined gradient of CrossEntropy + Softmax w.r.t logits.
        Simplifies cleanly to: (probs - y_true) / batch_size
        """
        probs      = _softmax(y_pred)
        y_oh       = _to_one_hot(y_true, y_pred.shape[1])
        batch_size = y_pred.shape[0]

        grad = (probs - y_oh) / batch_size             # ✅ correct combined gradient
        return grad