import numpy as np


# --------------------------------------------------
# Preprocessing helper
# --------------------------------------------------

def preprocess(images):
    """Flatten (N,28,28) uint8 → (N,784) float32 normalised to [0,1]."""
    N = images.shape[0]
    return images.reshape(N, -1).astype(np.float32) / 255.0


# --------------------------------------------------
# Keras loader with fallback
# --------------------------------------------------

def _keras_load(dataset):
    """Load raw arrays via Keras. Falls back to tensorflow.keras if needed."""
    try:
        if dataset == "mnist":
            from keras.datasets import mnist as ds
        else:
            from keras.datasets import fashion_mnist as ds
        return ds.load_data()
    except ImportError:
        pass

    try:
        if dataset == "mnist":
            from tensorflow.keras.datasets import mnist as ds
        else:
            from tensorflow.keras.datasets import fashion_mnist as ds
        return ds.load_data()
    except ImportError:
        raise ImportError("Neither 'keras' nor 'tensorflow' found. Please install one.")


# --------------------------------------------------
# Main load_data function
# --------------------------------------------------

def load_data(dataset="mnist"):
    """
    Load MNIST or Fashion-MNIST with a train/val/test split.

    Parameters
    ----------
    dataset : 'mnist' | 'fashion_mnist' | 'fashion'

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    X arrays : shape (N, 784), float32, values in [0, 1]
    y arrays : shape (N,),     int32  — raw integer labels
    """

    dataset = dataset.lower()
    if dataset == "fashion_mnist":
        dataset = "fashion"

    if dataset not in ("mnist", "fashion"):
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'mnist' or 'fashion_mnist'.")

    (X_train_full, y_train_full), (X_test, y_test) = _keras_load(dataset)

    # Preprocess
    X_train_full = preprocess(X_train_full)
    X_test       = preprocess(X_test)
    y_train_full = y_train_full.astype(np.int32)
    y_test       = y_test.astype(np.int32)

    # split last 10% of training data as validation set
    val_size = int(0.1 * len(X_train_full))
    X_val    = X_train_full[-val_size:]
    y_val    = y_train_full[-val_size:]
    X_train  = X_train_full[:-val_size]
    y_train  = y_train_full[:-val_size]

    print(
        f"[DataLoader] {dataset} — "
        f"train={X_train.shape[0]}  val={X_val.shape[0]}  test={X_test.shape[0]}"
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)