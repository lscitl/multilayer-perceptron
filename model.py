#!/usr/bin/python3

import sys
import shutil
import pandas as pd
import numpy as np
import copy
from enum import Enum, auto
from collections.abc import Iterable

from typing import Any, Iterable

# import pickle
# from load_csv import load, get_cur_dir

EPS = 1e-7

class LAYER(Enum):
    INPUT = auto()
    DENSE = auto()
    DROPOUT = auto()
    OUTPUT = auto()


class callback:
    """callbacks"""

    def __init__(self):
        pass


class EarlyStopping(callback):
    """EarlyStopping"""

    def __init__(self, monitor = "val_loss"):
        self.monitor = monitor


class layers:

    layer_to_str = {
        LAYER.INPUT:"Input",
        LAYER.DENSE:"Dense",
        LAYER.DROPOUT:"Dropout",
        LAYER.OUTPUT:"Output"
    }

    def __init__(self):
        """layer init."""
        self.type: LAYER = None
        self.shape = None
        self.activation = None
        self.input_dim = None
        self.weights_initializer = None

    def __init__(
        self,
        layer_type: LAYER,
        shape: int | tuple[int],
        activation: str,
        input_dim: int,
        weights_initializer: str
    ):
        """layer init."""
        self.type: LAYER = layer_type
        self.shape = shape
        self.activation = activation
        self.input_dim = input_dim
        self.weights_initializer = weights_initializer

    @staticmethod
    def Dense(
        shape: int,
        activation: str,
        input_dim: tuple = None,
        weights_initializer: str = None
    ):
        """Create dense layer."""

        return layers(LAYER.DENSE, shape, activation, input_dim, weights_initializer)

    @staticmethod
    def getLayer(layer: LAYER) -> str:
        return layers.layer_to_str[layer]

    def getLayer(self) -> str:
        return layers.layer_to_str[self.type]


class model:
    """
    A model for neural network
    """

    def __init__(self):
        self.layer: list[layers] = []
        self.weight: list[np.ndarray] = []
        self.bias: list[np.ndarray] = []
        self.input_dim: None
        self.output_dim: None

        self.iscompile = False
        self.optimizer = None
        self.loss = None
        self.metrics = None

    def _assert_compile(self):
        assert self.iscompile, "model should be compiled before train/test the model."

    def add(self, layer: layers):
        """add layer"""
        if len(self.layer) != 0:
            assert layer.input_dim is not None, "first layer should be INPUT layer."
            self.input_dim = layer.input_dim
            self.layer.append(layer)
        else:
            assert layer.type != LAYER.INPUT, "Input layer can be set only in the first layer."
            self.layer.append(layer)

    def createNetwork(self, layer: Iterable[layers]):
        """Create modle with given layer"""
        for l in layer:
            self.add(l)
        return self

    def compile(self, optimizer, loss, metrics):
        """set optimizer, loss, metrics"""

        assert self.iscompile is False, "model is already compiled."

        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.iscompile = True

    def fit(
        self,
        x=None,
        y=None,
        epochs=1,
        batch_size=32,
        callbacks: list[callback]=None
    ):
        """train data."""

        self._assert_compile()

    def evaluate(self,
                 x=None,
                 y=None):
        """evaluate the model."""

        self._assert_compile()

    def summary(self):
        """Show layer structure."""
        
        term_size = shutil.get_terminal_size() # -> 454
        print("     ")
        print("-" * 2)
        for l1, l2 in zip(self.layer, self.layer[1:]):
            print(f"{term_size}")

    def _init_params(self) -> None:

        self._layer_check()

        if len(self.weight) == 0:
            for l, l_next in zip(self.layer, self.layer[1:]):
                l.shape


    def _layer_check(self) -> None:

        assert len(self.layer) > 1, "check layers."
        assert self.layer[0].type == LAYER.INPUT, "first layer should be an input layer."

        for layer in self.layer[1:-1]:
            assert layer.type != LAYER.INPUT and layer.type != LAYER.OUTPUT, "hidden layer should not be an input or output layer."
        
        assert layer.type == LAYER.OUTPUT, "last layer sould be an output layer."


    def _linear_forward(self, ) -> None:
        return


    def _propagate(self, w: np.ndarray, b: np.ndarray,
                X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Args
            w: weights (n_category, n_feature)
            b: bias (n_category,)
            X: train data X (n_data, n_feature)
            Y: train data Y (n_data, n_category)

        Return
            dw: gradient loss of weights
            db: gradient loss of bias
            cost: negative log-likelihood cost for logistic regression
        """

        # m: data size(n_data)
        m = X.shape[0]

        # A: Predicted value(Y_hat). (n_data, n_category)
        A = sigmoid(X @ w.T + b)

        # Cost(loss) add epsilon for preventing errors
        cost = - np.sum(Y * np.log(A + EPS) + (1 - Y) * np.log(1 - A + EPS)) / m

        dw = (A - Y).T @ X / m
        db = np.sum((A - Y).T, axis=1) / m

        cost = np.squeeze(np.array(cost))

        return dw, db, cost


class Sequential:
    """
    Create model with input layers.
    """
    
    def __init__(self, layer: Iterable) -> model:
        """init with layers."""
        self.model = model()

        for l in layer:
            self.model.add(l)

    def __new__(cls):
        return cls.model


def sigmoid(z: np.ndarray) -> np.ndarray:
    """sigmoid function."""

    # for preventing overflow
    z_clipped = np.clip(z, -32, 32)
    return 1 / (np.exp(-z_clipped) + 1)


# def softmax(z: np.ndarray) -> np.ndarray:
#     """softmax function."""

#     exp_z = np.exp(z - np.max(z))
#     return exp_z / np.sum(exp_z, axis=0)


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





def GD_optimizer(w: np.ndarray, b: np.ndarray,
                 X: np.ndarray, Y: np.ndarray,
                 epoch: int = 1,
                 lr: float = 0.001,
                 print_cost: int = 0) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Args
        w: weights (n_category, n_feature)
        b: bias (n_category,)
        X: train data X (n_data, n_feature)
        Y: train data Y (n_data, n_category)
        epoch: number of iterations
        lr: learning rate
        print_cost: printing the loss every given steps
    
    Return
        tuple(w, b, costs)
        costs: cost list of every 100 steps
    """

    return BGD_optimizer(w, b, X, Y, epoch=epoch, batch=X.shape[0], lr=lr, print_cost=print_cost)


def SGD_optimizer(w: np.ndarray, b: np.ndarray,
                  X: np.ndarray, Y: np.ndarray,
                  epoch: int = 1,
                  lr: float = 0.001,
                  print_cost: int = 0) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Args
        w: weights (n_category, n_feature)
        b: bias (n_category,)
        X: train data X (n_data, n_feature)
        Y: train data Y (n_data, n_category)
        epoch: number of iterations
        lr: learning rate
        print_cost: printing the loss every given steps
    
    Return
        tuple(w, b, costs)
        costs: cost list of every 100 steps
    """

    return BGD_optimizer(w, b, X, Y, epoch=epoch, batch=1, lr=lr, print_cost=print_cost)


def BGD_optimizer(w: np.ndarray, b: np.ndarray,
                  X: np.ndarray, Y: np.ndarray,
                  epoch: int = 1,
                  batch: int = 32,
                  lr: float = 0.001,
                  print_cost: int = 0) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Args
        w: weights (n_category, n_feature)
        b: bias (n_category,)
        X: train data X (n_data, n_feature)
        Y: train data Y (n_data, n_category)
        epoch: number of iterations
        batch: data size for updating w, b
        lr: learning rate
        print_cost: printing the loss every given steps
    
    Return
        tuple(w, b, costs)
        costs: cost list of every 100 steps
    """

    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    lim = len(X) // batch + (1 if len(X) % batch else 0)

    for i in range(epoch):

        # batch loop
        for j in range(lim):
        
            # calculate dw, db, cost
            dw, db, cost = propagate(w, b, X[batch * j: batch * (j + 1)], Y[batch * j: batch * (j + 1)])

            # update w, b
            w = w - lr * dw
            b = b - lr * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % print_cost == 0:
            print (f"The cost of iteration {i}: {cost}")

        # # check converged
        # if (np.isclose(dw, np.zeros_like(dw), rtol=1e-6).all() or
        #     np.isclose(db, np.zeros_like(db), rtol=1e-6).all()):
        #     break

    costs.append(cost)

    return w, b, costs


def predict(w: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Args
        w: weights (n_category, n_feature)
        b: bias (n_category,)
        X: train data X (n_data, n_feature)

    Return
        Y_pred: predict value (n_data, n_category)
    """

    m = X.shape[0]

    A = sigmoid(X @ w.T + b)

    if w.shape[0] == 1:
        Y_pred = (A > 0.5).astype(int)
        # for i in range(A.shape[1]):
        #     Y_pred[i, 0] = 1 if A[i, 0] > 0.5 else 0
    else:
        Y_pred = get_one_hot_value(A)

    return Y_pred.astype(int)


def fit(x_train: np.ndarray, y_train: np.ndarray,
          w_init: np.ndarray = None, b_init: np.ndarray = None,
          epoch: int = 1,
          batch: int = 32,
          lr: int = 0.001,
          print_cost: int = 100,
          optimizer: str = "BGD") -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Args
        x_train: train data for input (n_data, n_feature)
        y_train: train data for output (n_data, n_category)
        w_init: init w (n_category, n_feature)
        b_init: inti bias (n_category,)
        epoch: number of iterations
        batch: data size for updating w, b
        lr: learning rate
        print_cost: printing the cost every n steps. 0 or None for not printing.
    
    Return
        tuple(w, b, costs)
        costs: cost list of every 100 steps
    """

    assert len(y_train.shape) in (1, 2), "invalid y value."
    assert optimizer in ["BGD", "SGD", "GD"], "invalid optimizer."

    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)

    n_feature = x_train.shape[1]
    n_category = y_train.shape[1]

    # w shape is (n_category, n_feature), and b shape is (n_category,)
    if w_init is None:
        w_init = np.zeros((n_category, n_feature))
    if b_init is None:
        b_init = np.zeros(n_category)

    match optimizer:
        case "BGD":
            w, b, costs = BGD_optimizer(w_init, b_init, x_train, y_train, epoch, batch, lr, print_cost)
        case "SGD":
            w, b, costs = SGD_optimizer(w_init, b_init, x_train, y_train, epoch, lr, print_cost)
        case "GD":
            w, b, costs = GD_optimizer(w_init, b_init, x_train, y_train, epoch, lr, print_cost)

    return w, b, costs



def evaluate(w: np.ndarray, b: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """return accuracy of prediction."""
    
    y_pred = predict(w, b, x_test)
    
    assert y_pred.shape == y_test.shape, "invalid input. different shape"

    wrong = np.sum(np.abs(y_pred - y_test) / 2)
    
    return (1 - wrong / len(y_test)) * 100





if __name__ == "__main__":

    print(model.__doc__)
