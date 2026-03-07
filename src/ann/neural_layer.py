import numpy as np


class Dense:
    """
    Fully connected (dense) layer.
    Performs: Z = XW + b
    """

    def __init__(self, input_dim, output_dim, weight_init="random", weight_decay=0.0):
        """
        input_dim   : number of input features
        output_dim  : number of neurons in this layer
        weight_init : 'random' | 'xavier' | 'zeros'
        weight_decay: L2 regularisation coefficient
        """

        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.weight_decay = weight_decay

        # Initialize weights
        if weight_init == "random":
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        elif weight_init == "xavier":
            # ✅ FIX: Glorot uniform (correct formula)
            limit = np.sqrt(6.0 / (input_dim + output_dim))
            self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))

        elif weight_init == "zeros":
            self.W = np.zeros((input_dim, output_dim))

        else:
            raise ValueError("weight_init must be 'random', 'xavier', or 'zeros'")

        # Initialize bias to zero
        self.b = np.zeros((1, output_dim))

        # Placeholders
        self.X      = None
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, X):
        """
        X shape: (batch_size, input_dim)
        Returns: (batch_size, output_dim)
        """
        self.X = X
        Z = np.dot(X, self.W) + self.b
        return Z

    def backward(self, dZ):
        """
        dZ shape: (batch_size, output_dim)
        Returns dX shape: (batch_size, input_dim)
        """
        batch_size = self.X.shape[0]

        # ✅ FIX: divide by batch_size for correct gradient scale
        self.grad_W = np.dot(self.X.T, dZ) / batch_size

        # ✅ FIX: use mean instead of sum
        self.grad_b = np.mean(dZ, axis=0, keepdims=True)

        # ✅ FIX: add L2 regularisation term inside backward
        if self.weight_decay > 0.0:
            self.grad_W += self.weight_decay * self.W

        # Gradient w.r.t input to pass to previous layer
        dX = np.dot(dZ, self.W.T)

        return dX