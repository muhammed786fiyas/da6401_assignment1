# DA6401 Assignment 1 — Neural Network from Scratch

A fully **NumPy-based** feedforward neural network trained on **MNIST** and **Fashion-MNIST**, built from scratch without any deep learning frameworks. Experiment tracking via **Weights & Biases**.

**W&B Report:** https://wandb.ai/muhammed786fiyas-iit-madras/da6401-a1/reports/DA6401-Assignment-1-MLP-on-MNIST--VmlldzoxNjEzNDg1MQ

---

## Results

| Dataset | Optimizer | Activation | Layers | Hidden Size | Test Accuracy | Test F1 |
|---------|-----------|------------|--------|-------------|---------------|---------|
| MNIST | Adam | ReLU | 3 | 128×128×128 | **97.68%** | **0.9765** |
| Fashion-MNIST | Adam | ReLU | 3 | 128×128×128 | 87.60% | 0.876 |

Best config: `adam`, `lr=0.001`, `batch_size=32`, `3 hidden layers of 128`, `relu`, `xavier init`, `15 epochs`

---

## Project Structure

```
submission_1/
├── models/
│   ├── best_model.npy            # Best model weights
│   └── best_config.json          # Best model hyperparameters
├── notebooks/
│   └── data_exploration.ipynb
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py        # Sigmoid, Tanh, ReLU
│   │   ├── neural_layer.py       # Dense layer (forward/backward)
│   │   ├── neural_network.py     # Full model (forward/backward/evaluate/save/load)
│   │   ├── objective_functions.py# CrossEntropy, MeanSquaredError
│   │   └── optimizers.py         # SGD, Momentum, NAG, RMSProp, Adam, Nadam
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py        # MNIST/Fashion-MNIST loading & preprocessing
│   ├── best_model.npy            # Best model weights (copy)
│   ├── best_config.json          # Best model config (copy)
│   ├── inference.py              # Load and evaluate saved model
│   └── train.py                  # Train from CLI
├── README.md
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `keras` (data loading only), `scikit-learn`, `wandb`, `matplotlib`

---

## Training

```bash
cd src
python train.py \
  -d mnist \
  -e 15 \
  -b 32 \
  -l cross_entropy \
  -o adam \
  -lr 0.001 \
  -wd 0.0 \
  -nhl 3 \
  -sz 128 128 128 \
  -a relu \
  -wi xavier \
  --use_wandb \
  --wandb_project da6401-a1
```

### CLI Arguments

| Flag | Long form | Default | Description |
|------|-----------|---------|-------------|
| `-d` | `--dataset` | `mnist` | `mnist` or `fashion_mnist` |
| `-e` | `--epochs` | `10` | Number of training epochs |
| `-b` | `--batch_size` | `32` | Mini-batch size |
| `-l` | `--loss` | `cross_entropy` | `cross_entropy` or `mean_squared_error` |
| `-o` | `--optimizer` | `adam` | `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` |
| `-lr` | `--learning_rate` | `0.001` | Initial learning rate |
| `-wd` | `--weight_decay` | `0.0` | L2 regularisation coefficient |
| `-nhl` | `--num_layers` | `3` | Number of hidden layers |
| `-sz` | `--hidden_size` | `128 128 128` | Neurons per hidden layer (space separated) |
| `-a` | `--activation` | `relu` | `relu`, `sigmoid`, `tanh` |
| `-wi` | `--weight_init` | `xavier` | `xavier`, `random`, or `zeros` |
| `--use_wandb` | | `False` | Enable W&B logging |
| `--wandb_project` | | `da6401-a1` | W&B project name |

---

## Inference

```bash
cd src
python inference.py -d mnist
```

Automatically finds `best_model.npy` and `best_config.json` from `src/` or `models/`.
Outputs **Accuracy**, **Precision**, **Recall**, and **F1-score** to stdout.

---

## Design Notes

### Forward Pass
Each `Dense.forward(X)` computes `z = X @ W + b` and passes through the chosen activation. The output layer uses no activation (raw logits). Softmax is applied inside the loss function.

### Backward Pass
`NeuralNetwork.backward()` chains `Dense.backward()` calls in reverse order. Each layer stores `self.grad_W` and `self.grad_b` after every backward call.

### Gradient Layout
`grad_W[0]` / `grad_b[0]` → **last (output) layer**
`grad_W[-1]` / `grad_b[-1]` → **first hidden layer**

### Weight Initialization
- `xavier` — uniform distribution scaled by `sqrt(6 / (fan_in + fan_out))`
- `random` — standard normal distribution
- `zeros` — all zeros (demonstrates symmetry problem)

### Optimizers Implemented
SGD, Momentum, NAG (Nesterov Accelerated Gradient), RMSProp, Adam, Nadam — all from scratch in NumPy.

---

## W&B Experiments

| Section | Experiment | Key Finding |
|---------|------------|-------------|
| 2.1 | Data Exploration | 10 classes, visually similar pairs: 4↔9, 3↔8, 1↔7 |
| 2.2 | Hyperparameter Sweep (127 runs) | Best: adam/nadam + relu + lr=0.001 |
| 2.3 | Optimizer Comparison | Nadam/Adam F1=0.977 vs SGD F1=0.852 |
| 2.4 | Vanishing Gradient | Sigmoid grad norm ~0.0007 vs ReLU ~0.008 |
| 2.5 | Dead Neurons | ReLU lr=0.1 → 122/128 neurons dead from epoch 1 |
| 2.6 | Loss Functions | Cross-entropy converges faster than MSE |
| 2.7 | Global Performance | Best test accuracy 97.68% |
| 2.8 | Confusion Matrix | Top errors: 9↔4, 7↔2, 5↔3 |
| 2.9 | Weight Init | Zeros init stuck at 10.5% accuracy all epochs |
| 2.10 | Fashion-MNIST | Best config achieves F1=0.876 vs 0.977 on MNIST |