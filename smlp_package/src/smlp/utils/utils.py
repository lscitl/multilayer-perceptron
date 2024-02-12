import numpy as np
import pandas as pd


def get_one_hot_value(x: np.ndarray) -> np.ndarray:
    """one hot value for categorical. Max value -> 1, and the others -> 0."""

    max_idx = np.argmax(x, axis=1)
    res = np.zeros_like(x)
    for i in range(len(max_idx)):
        res[i, max_idx[i]] = 1
    return res


def one_hot_encoding(x: pd.Series) -> np.ndarray:
    """convert categorical variable to binary vector."""

    unique_val = x.unique()
    unique_val.sort()
    return (x.values.reshape(-1, 1) == unique_val).astype(int)
