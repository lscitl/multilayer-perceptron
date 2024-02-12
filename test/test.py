from typing import overload
from functools import partial
import numpy as np


def func(a: np.ndarray):
    a = a - a.mean(axis=0)
    return a


if __name__ == "__main__":
    a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])

    b = a[:3]

    b[0][0] = 10

    print(a)
