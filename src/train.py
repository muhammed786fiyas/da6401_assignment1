import argparse
import numpy as np
import os
import json
import sys

from sklearn.metrics import accuracy_score, f1_score

from utils.data_loader import load_data
from ann.neural_layer import Dense
from ann.activations import ReLU, Sigmoid, Tanh
from ann.objective_functions import MeanSquaredError, CrossEntropy
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD, Momentum, NAG, RMSProp, Adam, Nadam


# --------------------------------------------------
# Model Construction
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

    model.add(Dense(prev_dim, 10, weight_init))

    return model


# --------------------------------------------------
# Optimizer Selection
# --------------------------------------------------

def get_optimizer(name, learning_rate):

    if name == "sgd":
        return SGD(learning_rate)
    elif name == "momentum":
        return Momentum(learning_rate)
    elif name == "nag":
        return NAG(learning_rate)
    elif name == "rmsprop":
        return RMSProp(learning_rate)
    elif name == "adam":
        return Adam(learning_rate)
    else:
        return Nadam(learning_rate)


# --------------------------------------------------
# Training Function
# --------------------------------------------------

def train(args):

    np.random.seed(42)

    wandb_run = None
    if getattr(args, "use_wandb", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=getattr(args, "wandb_project", "da6401-assignment-1"),
                config=vars(args),
                name=f"{args.optimizer}_lr{args.learning_rate}_layers{args.hidden_size}"
            )
        except ImportError:
            print("[WARNING] wandb not installed — skipping W&B logging.")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(args.dataset)

    # Build model
    model = build_model(
        input_dim=784,
        hidden_sizes=args.hidden_size,
        activation_name=args.activation,
        weight_init=args.weight_init
    )

    if args.loss == "cross_entropy":
        loss_fn = CrossEntropy()
    else:
        loss_fn = MeanSquaredError()

    model.set_loss(loss_fn)  

    optimizer = get_optimizer(args.optimizer, args.learning_rate)

    best_accuracy = 0

    for epoch in range(args.epochs):

        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss  = 0
        num_batches = 0

        for i in range(0, len(X_shuffled), args.batch_size):

            X_batch = X_shuffled[i:i + args.batch_size]
            y_batch = y_shuffled[i:i + args.batch_size]

            # Forward pass
            y_pred = model.forward(X_batch)

            # Compute loss
            loss = loss_fn.forward(y_pred, y_batch)
            epoch_loss  += loss
            num_batches += 1

            model.backward(y_batch, y_pred)

            # Update parameters
            optimizer.step(model.get_trainable_layers())

        epoch_loss /= num_batches

        # Evaluate on validation set
        val_pred    = model.forward(X_val)
        val_labels  = np.argmax(val_pred, axis=1)
        val_acc     = accuracy_score(y_val, val_labels)
        val_f1      = f1_score(y_val, val_labels, average="macro", zero_division=0)

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")

        if wandb_run is not None:
            wandb_run.log({
                "epoch":    epoch + 1,
                "loss":     epoch_loss,
                "val_acc":  val_acc,
                "val_f1":   val_f1,
            })

        # # Save best model by accuracy
        # if val_acc > best_accuracy:
        #     best_accuracy = val_acc

        #     src_dir    = os.path.dirname(os.path.abspath(__file__))
        #     models_dir = os.path.join(src_dir, '..', 'models')
        #     os.makedirs(models_dir, exist_ok=True)

        #     weights = model.get_weights()

        #     np.save(os.path.join(src_dir, 'best_model.npy'), weights)
        #     np.save(os.path.join(models_dir, 'best_model.npy'), weights)

        #     with open(os.path.join(src_dir, 'best_config.json'), 'w') as f:
        #         json.dump(vars(args), f, indent=2)
        #     with open(os.path.join(models_dir, 'best_config.json'), 'w') as f:
        #         json.dump(vars(args), f, indent=2)

        #     print(f"  → Best model saved (val_acc={val_acc:.4f})")

    # Final test evaluation
    test_pred   = model.forward(X_test)
    test_labels = np.argmax(test_pred, axis=1)
    test_acc    = accuracy_score(y_test, test_labels)
    test_f1     = f1_score(y_test, test_labels, average="macro", zero_division=0)

    print(f"\nTest Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    if wandb_run is not None:
        wandb_run.log({"test_accuracy": test_acc, "test_f1": test_f1})
        wandb_run.finish()

    return test_acc, test_f1


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d",   "--dataset",       choices=["mnist", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("-e",   "--epochs",         type=int,   default=10)
    parser.add_argument("-b",   "--batch_size",     type=int,   default=32)
    parser.add_argument("-l",   "--loss",           choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o",   "--optimizer",      choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="adam")
    parser.add_argument("-lr",  "--learning_rate",  type=float, default=0.001)
    parser.add_argument("-wd",  "--weight_decay",   type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers",     type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",    type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("-a",   "--activation",     choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("-wi",  "--weight_init",    choices=["random", "xavier", "zeros"], default="xavier")
    parser.add_argument("--use_wandb",              action="store_true")
    parser.add_argument("--wandb_project",          type=str, default="da6401-assignment-1")

    args = parser.parse_args()

    train(args)