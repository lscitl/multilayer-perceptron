import numpy as np


def relu(x: np.ndarray, max_val=None):
    """relu function."""
    if max_val is not None:
        return np.minimum(max_val, np.maximum(0, x))
    return np.maximum(0, x)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """sigmoid function."""

    # for preventing overflow
    z_clipped = np.clip(z, -32, 32)
    return 1 / (np.exp(-z_clipped) + 1)


def softmax(z: np.ndarray) -> np.ndarray:
    """softmax function."""

    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
