import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.data_loader import load_data
from ann.neural_layer import Dense
from ann.activations import ReLU, Sigmoid, Tanh
from ann.neural_network import NeuralNetwork
from ann.objective_functions import CrossEntropy, MeanSquaredError


# --------------------------------------------------
# Model Builder
# --------------------------------------------------

def build_model(input_dim, hidden_sizes, activation_name, weight_init):

    model = NeuralNetwork()

    if activation_name == "relu":
        activation = ReLU
    elif activation_name == "sigmoid":
        activation = Sigmoid
    else:
        activation = Tanh

    prev_dim = input_dim

    for size in hidden_sizes:
        model.add(Dense(prev_dim, size, weight_init))
        model.add(activation())
        prev_dim = size

    # ✅ FIX: no Softmax at end — softmax is applied inside objective_functions
    model.add(Dense(prev_dim, 10, weight_init))

    return model


# --------------------------------------------------
# Load Saved Weights
# --------------------------------------------------

def load_weights(model, weights_path):
    weights = np.load(weights_path, allow_pickle=True).item()
    trainable_layers = model.get_trainable_layers()
    for i, layer in enumerate(trainable_layers):
        layer.W = weights[f"W{i}"]
        layer.b = weights[f"b{i}"]


# --------------------------------------------------
# Find config file
# --------------------------------------------------

def find_config(model_path, config_path=None):
    """Search multiple locations for best_config.json."""
    if config_path and os.path.exists(config_path):
        return config_path

    # ✅ FIX: search multiple possible paths instead of hardcoding
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(model_path)), "best_config.json"),
        "models/best_config.json",
        "../models/best_config.json",
        "src/best_config.json",
        "best_config.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find best_config.json. Tried: {candidates}"
    )


# --------------------------------------------------
# Inference Function
# --------------------------------------------------

def run_inference(args):

    # Load config
    config_path = find_config(args.model_path, getattr(args, "config_path", None))
    with open(config_path, "r") as f:
        config = json.load(f)

    # ✅ FIX: use new 3-tuple return format from load_data
    _, _, (X_test, y_test) = load_data(args.dataset)

    # Rebuild model from saved config
    hidden_sizes = config["hidden_size"]
    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes] * config.get("num_layers", 3)

    model = build_model(
        input_dim=784,
        hidden_sizes=hidden_sizes,
        activation_name=config["activation"],
        weight_init=config["weight_init"]
    )

    # Load weights
    load_weights(model, args.model_path)

    # Forward pass — outputs raw logits
    logits = model.forward(X_test)

    # ✅ FIX: apply softmax manually here for inference (not via loss fn)
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits     = np.exp(logits_shifted)
    probs          = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    predictions = np.argmax(probs, axis=1)

    # Metrics
    acc       = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro", zero_division=0)
    recall    = recall_score(y_test, predictions, average="macro", zero_division=0)
    f1        = f1_score(y_test, predictions, average="macro", zero_division=0)

    print("\n── Inference Results ─────────────────────────────")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Model     : {args.model_path}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("──────────────────────────────────────────────────\n")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("-d",  "--dataset",     default="mnist",
                        choices=["mnist", "fashion_mnist", "fashion"])
    parser.add_argument("--model_path",         default="models/best_model.npy")
    parser.add_argument("--config_path",        default=None)
    parser.add_argument("--use_wandb",          action="store_true")
    parser.add_argument("--wandb_project",      default="da6401-assignment-1")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_inference(args)