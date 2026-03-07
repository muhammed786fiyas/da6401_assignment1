import numpy as np
from ann.objective_functions import CrossEntropy, MeanSquaredError


class NeuralNetwork:

    def __init__(self, args=None):

        self.layers = []
        self.loss_fn = None

        if args is not None and hasattr(args, "hidden_size"):

            from ann.neural_layer import Dense
            from ann.activations import ReLU, Sigmoid, Tanh

            input_dim = 784
            hidden_sizes = args.hidden_size
            activation = args.activation
            weight_init = args.weight_init

            # choose activation
            if activation == "relu":
                act = ReLU
            elif activation == "sigmoid":
                act = Sigmoid
            else:
                act = Tanh

            prev_dim = input_dim

            for size in hidden_sizes:
                self.layers.append(Dense(prev_dim, size, weight_init))
                self.layers.append(act())
                prev_dim = size

            self.layers.append(Dense(prev_dim, 10, weight_init))

            # store loss function
            if hasattr(args, "loss"):
                if args.loss == "cross_entropy":
                    self.loss_fn = CrossEntropy()
                else:
                    self.loss_fn = MeanSquaredError()

    # --------------------------------------------------
    # Add Layer
    # --------------------------------------------------

    def add(self, layer):
        self.layers.append(layer)

    # --------------------------------------------------
    # Set Loss Function
    # --------------------------------------------------

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    # --------------------------------------------------
    # Forward Pass
    # --------------------------------------------------

    def forward(self, X):

        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output, output

    # --------------------------------------------------
    # Backward Pass
    # --------------------------------------------------

    def backward(self, y_true, y_pred):
        if isinstance(y_pred, tuple): y_pred = y_pred[0]


        if self.loss_fn is not None:
            gradient = self.loss_fn.backward(y_pred, y_true)
        else:
            gradient = y_pred - y_true

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        # collect gradients from trainable layers
        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):
            if hasattr(layer, "grad_W"):
                grad_W_list.append(layer.grad_W)
                grad_b_list.append(layer.grad_b)

        grad_W = np.empty(len(grad_W_list), dtype=object)
        grad_b = np.empty(len(grad_b_list), dtype=object)

        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            grad_W[i] = gw
            grad_b[i] = gb

        return grad_W, grad_b   

    # --------------------------------------------------
    # Get Trainable Layers
    # --------------------------------------------------

    def get_trainable_layers(self):

        trainable = []

        for layer in self.layers:
            if hasattr(layer, "W"):
                trainable.append(layer)

        return trainable

    # --------------------------------------------------
    # Set Weights
    # --------------------------------------------------

    def set_weights(self, weights):

        trainable_layers = self.get_trainable_layers()

        if hasattr(weights, "__dict__"):
            weights = vars(weights)

        for i, layer in enumerate(trainable_layers):
            layer.W = weights[f"W{i}"]
            layer.b = weights[f"b{i}"]

    # --------------------------------------------------
    # Get Weights
    # --------------------------------------------------

    def get_weights(self):

        weights = {}

        trainable_layers = self.get_trainable_layers()

        for i, layer in enumerate(trainable_layers):
            weights[f"W{i}"] = layer.W
            weights[f"b{i}"] = layer.b

        return weights
    
    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    def evaluate(self, X, y):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        logits, _   = self.forward(X)
        y_pred   = np.argmax(logits, axis=1)
        y_true   = y.astype(int)

        accuracy  = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # also compute loss if loss_fn available
        loss = 0.0
        if self.loss_fn is not None:
            loss = float(self.loss_fn.forward(logits, y))

        return {
            "loss":      loss,
            "accuracy":  float(accuracy),
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "logits":    logits,
        }