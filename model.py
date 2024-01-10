#!/usr/bin/python3

import sys
import shutil
import time
import pandas as pd
import numpy as np
import copy
from enum import Enum, auto
from collections.abc import Iterable
from optimizer import Optimizer, Adam, SGD

from typing import Any, Iterable, Callable
from functools import partial

# import pickle
# from load_csv import load, get_cur_dir

EPS = 1e-7

class LAYER(Enum):
    INPUT = auto()
    DENSE = auto()
    DROPOUT = auto()


class Callback:
    """callbacks"""

    def __init__(self):
        pass


class EarlyStopping(Callback):
    """EarlyStopping"""

    def __init__(self, monitor = "val_loss"):
        self.monitor = monitor

class Initializer:
    """Initalizer"""

    @staticmethod
    def glorotNormal(shape: tuple):
        """Glorot / Xavier normal initializer."""
        stddev = np.sqrt(2.0 / (shape[0] + shape[1]))
        return np.random.normal(0, stddev, shape)

    @staticmethod
    def glorotUniform(shape: tuple):
        """Glorot / Xavier uniform initializer."""
        limit = np.sqrt(6.0 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    def heNormal(shape: tuple):
        """He normal initializer."""
        stddev = np.sqrt(2.0 / shape[0])
        return np.random.normal(0, stddev, shape)

    @staticmethod
    def heUniform(shape: tuple):
        """He uniform initializer."""
        limit = np.sqrt(6.0 / shape[0])
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    def zeros(shape: tuple):
        """Zero initializer. For bias."""
        return np.zeros(shape)


class Layers:
    """
    layer class
    
    """

    layer_to_str = {
        LAYER.INPUT:"Input",
        LAYER.DENSE:"Dense",
        LAYER.DROPOUT:"Dropout",
    }

    def __init__(
        self,
        layer_type: LAYER,
        layer_dim: int,
        activation: str,
        weights_initializer: str | Callable
    ):
        """layer init."""
        self.type: LAYER = layer_type
        self.layer_dim = layer_dim
        self.activation = activation

        if layer_type != LAYER.INPUT:
            if isinstance(weights_initializer, str):
                match weights_initializer:
                    case "heNormal":
                        self.weights_initializer = Initializer.heNormal
                    case "heUniform":
                        self.weights_initializer = Initializer.heUniform
                    case "glorotNormal" | "xavierNormal":
                        self.weights_initializer = Initializer.glorotNormal
                    case "glorotUniform" | "xavierUniform":
                        self.weights_initializer = Initializer.glorotUniform
                    case "zero":
                        self.weights_initializer = Initializer.zeros
                    case _:
                        raise AssertionError("Invalid initializer.")
            
            elif callable(weights_initializer):
                self.weights_initializer = weights_initializer

            else:
                raise AssertionError("Invalid initializer.")

    @staticmethod
    def Input(
        layer_dim: int
    ):
        """Create dense layer."""

        return Layers(LAYER.INPUT, layer_dim, None, None)

    @staticmethod
    def Dense(
        layer_dim: int,
        activation: str | Callable,
        weights_initializer: str="glorotNormal"
    ):
        """Create dense layer."""

        return Layers(LAYER.DENSE, layer_dim, activation, weights_initializer)

    # @staticmethod
    # def Dropout(

    # ):
    #     """Create dropout layer."""


    @staticmethod
    def getLayer(layer: LAYER) -> str:
        return Layers.layer_to_str[layer]

    def getLayer(self) -> str:
        return self.layer_to_str[self.type]



class Model:
    """
    A model for neural network
    """

    def __init__(self):
        self.layer: list[Layers] = []
        self.params: dict = {}
        self.grads: dict = {}

        self.iscompile = False
        self.optimizer: Optimizer = None
        self.loss = None
        self.metrics = None

    def _assert_compile(self):
        """compile checker."""
        assert self.iscompile, "model should be compiled before train/test the model."

    def _print_params(self):
        """print_params for debug."""

        for key in self.params.keys():
            print(key + ": ", self.params[key])

    def add(self, layer: Layers):
        """add layer"""

        assert isinstance(layer, Layers), "invalid layer type."

        if len(self.layer) == 0:
            assert layer.type == LAYER.INPUT, "first layer should be INPUT layer."
            self.layer.append(layer)

        else:
            assert layer.type != LAYER.INPUT, "Input layer can be set only in the first layer."
            self.layer.append(layer)

    @staticmethod
    def sequential(layer: Iterable):
        """Create modle with given layer"""

        model = Model()
        
        for l in layer:
            assert isinstance(l, Layers), "invalid layer is included."
            model.add(l)
        return model

    def compile(self, optimizer, loss, metrics):
        """set optimizer, loss, metrics"""

        assert self.iscompile is False, "model is already compiled."

        self._layer_check()
        self._init_params()

        # self._print_params()

        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            match optimizer:
                case "SGD":
                    self.optimizer = SGD()
                case "adam":
                    self.optimizer = Adam()
                case _:
                    raise AssertionError("Invalid optimizer.")
        else:
            raise AssertionError("Optimizer should be string or Optimizer class.")
            
        self.loss = loss
        self.metrics = metrics
        self.iscompile = True

    def fit(
        self,
        x=None,
        y=None,
        epochs=1,
        batch_size=32,
        callbacks: list[Callback]=None
    ):
        """train data."""

        self._assert_compile()

        history = {}
        history["loss"] = []
        
        batch_max = x.shape[0] // batch_size
        seed = int(time.time() * 1000000)
        seed = 10
    
        for i in range(epochs):
            start_time = time.time()
            loss_tmp = []

            # for mini-batch
            np.random.seed(seed)
            permutation = list(np.random.permutation(x.shape[0]))
            shuffled_x = x[permutation, :]
            shuffled_y = y[permutation, :]

            for n_batch in range(batch_max):
                mini_batch_x = shuffled_x[n_batch * batch_size: (n_batch + 1) * batch_size]
                mini_batch_y = shuffled_y[n_batch * batch_size: (n_batch + 1) * batch_size]
                # print(n_batch * batch_size, (n_batch + 1) * batch_size)

                AL, caches = self._model_forward(mini_batch_x)
                loss_tmp.append(self._compute_cost(AL, mini_batch_y))
                self._model_backward_and_update_params(AL, mini_batch_y, caches)

            seed += 1

            # for callback in callbacks:
            #     callback
            
            history['loss'].append(np.mean(loss_tmp))

            end_time = time.time()
            print_str = f"{i+1}/{epochs} - {int((end_time - start_time) * 1000)}ms/step"
            print(print_str)

        return history

    def evaluate(self,
                 x=None,
                 y=None):
        """evaluate the model."""

        self._assert_compile()

    def predict(self,
                x=None):
        """predict."""

        self._assert_compile()


    def summary(self):
        """Show layer structure."""
        
        term_size = shutil.get_terminal_size() # -> 454
        print("     ")
        print("-" * 2)
        for l1, l2 in zip(self.layer, self.layer[1:]):
            print(f"{term_size}")


    def _init_params(self) -> None:

        for i, (l, l_next) in enumerate(zip(self.layer, self.layer[1:]), start=1):
            self.params['W' + str(i)] = l_next.weights_initializer((l.layer_dim, l_next.layer_dim))
            self.params['b' + str(i)] = Initializer.zeros((1, l_next.layer_dim))

    def _layer_check(self) -> None:

        assert len(self.layer) > 1, "check layers."
        assert self.layer[0].type == LAYER.INPUT, "first layer should be an input layer."

        for layer in self.layer[1:-1]:
            assert layer.type != LAYER.INPUT, "hidden layer should not be an input layer."

    def _linear_forward(self, A, W, b) -> None:
        """
        Args
            A: activation from previous layer (batch_size, n_input)
            W: weights (n_input, n_output)
            b: bias (1, n_output)

        Return
            Z: calculated value
            cache: saved data of A, W, b for backward propagation.
        """

        # print("A, W shape:", A.shape, W.shape)

        Z = A @ W + b

        return Z, (A, W, b)

    def _linear_activation_forward(self, A_prev, W, b, activation):
        """
        Args
            A_prev: activation from previous layer
            W: weights
            b: bias
            activation: activation function

        Return
            A: the output of the activation function
            cache: linear_cache, activation cache
        """

        match activation:
            case "relu":
                Z, linear_cache = self._linear_forward(A_prev, W, b)
                A = relu(Z)
            case "sigmoid":
                Z, linear_cache = self._linear_forward(A_prev, W, b)
                A = sigmoid(Z)
            case "softmax":
                Z, linear_cache = self._linear_forward(A_prev, W, b)
                A = softmax(Z)
            case _:
                raise AssertionError("Invalid activation function.")

        # originally activation cache is the input of activation function.    
        # but in this code, activation cache is the result of activation function.
        activation_cache = A
        cache = (linear_cache, activation_cache)
        return A, cache

    def _model_forward(self, X):
        """
        Args
            X: input data
   
        Return
            A: the output of the last layer
            caches: cache values from all layers
        """

        caches = []
        A = X

        for i, layer in enumerate(self.layer[1:], start=1):
            A_prev = A

            A, cache = self._linear_activation_forward(A_prev, self.params['W' + str(i)], self.params['b' + str(i)], layer.activation)
            caches.append(cache)
        
        return A, caches
    
    def _compute_cost(self, AL: np.ndarray, Y: np.ndarray):
        """
        Args
            AL: the output of the last layer
            Y: the answer of input data
   
        Return
            cost: cost value from the loss function
        """

        m = Y.shape[0]

        match self.loss:
            case "binaryCrossentropy":
                cost = - np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m
            case _:
                raise AssertionError("Invalid loss function.")
        
        cost = np.squeeze(cost)

        return cost

    def _relu_backward(self, dA: np.ndarray, cache: np.ndarray):
        """
        Args
            dA: gradient of the cost w.r.t. relu output
            cache: cached data(previous input) of relu
        
        Return
            dZ: gradient of the cost w.r.t. linear output
        """

        assert dA.shape == cache.shape, "dA and cache shape should be the same."

        return cache * dA

    def _sigmoid_backward(self, dA: np.ndarray, cache: np.ndarray):
        """
        Args
            dA: gradient of the cost w.r.t. sigmoid output
            cache: cached data(previous input) of sigmoid
        
        Return
            dZ: gradient of the cost w.r.t. linear output
        """

        assert dA.shape == cache.shape, "dA and cache shape should be the same."

        return dA * cache * (1 - cache)

    def _softmax_backward(self, dA: np.ndarray, cache: np.ndarray):
        """
        Args
            dA: gradient of the cost w.r.t. softmax output
            cache: cached data(previous input) of softmax
        
        Return
            dZ: gradient of the cost w.r.t. linear output
        """

        assert dA.shape == cache.shape, "dA and cache shape should be the same."

        return dA * cache * (1 - cache)

    def _linear_backward(self, dZ: np.ndarray, cache: tuple[np.ndarray, np.ndarray, np.ndarray]):
        """
        Args
            dZ: gradient of the cost w.r.t. linear output
            cache: cached data(A_prev, W, b) of previous layer
        
        Return
            dA_prev: gradient of the cost w.r.t. the previous activation
            dW: gradient of the cost w.r.t. W
            db: gradient of the cost w.r.t. b
        """
        A_prev, W, _ = cache
        m = A_prev.shape[0]

        dW = (A_prev.T @ dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = dZ @ W.T
        
        return dA_prev, dW, db

    def _linear_activation_backward(self, dA, cache, activation):
        """
        Args
            dA: gradient of the cost w.r.t. layer
            cache: cached data(A_prev, W, b) of previous layer.
            activation: the layer's activation function
        
        Return
            dA_prev: gradient of the cost w.r.t. the previous activation
            dW: gradient of the cost w.r.t. W
            db: gradient of the cost w.r.t. b
        """
        linear_cache, activation_cache = cache
        
        match activation:
            case "relu":
                dZ = self._relu_backward(dA, activation_cache)
                
            case "sigmoid":
                dZ = self._sigmoid_backward(dA, activation_cache)
                
            case "softmax":
                dZ = self._softmax_backward(dA, activation_cache)

            case _:
                raise AssertionError("Invalid activation function.")
            
        dA_prev, dW, db = self._linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def _model_backward_and_update_params(self, AL: np.ndarray, Y: np.ndarray, caches: list):
        """
        Args
            AL: the output of the last layer
            Y: the answer vector of input data
            caches: linear and activation cache from the each layers

        """

        # m = AL.shape[1]

        match self.loss:
            case "binaryCrossentropy":
                dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
            case "mse":
                dAL = - 2 * (Y - AL)
            case _:
                raise AssertionError("Invalid loss function.")


        for i, layer in reversed(list(enumerate(self.layer[1:], start=0))):
            cache = caches[i]
            dAL, dW_tmp, db_tmp = self._linear_activation_backward(dAL, cache, layer.activation)
            self.grads["dA" + str(i)] = dAL
            self.grads["dW" + str(i + 1)] = dW_tmp
            self.grads["db" + str(i + 1)] = db_tmp

            # # self._update_param()

            # # # parameters will updated by the optimizer.
            # # self.params["W" + str(i + 1)] = self.optimizer.update(self.params["W" + str(i + 1)], dW_tmp)
            # # self.params["b" + str(i + 1)] = self.optimizer.update(self.params["b" + str(i + 1)], db_tmp)
            # self.params["W" + str(i + 1)] = self.params["W" + str(i + 1)] - self.optimizer.learning_rate * dW_tmp
            # self.params["b" + str(i + 1)] = self.params["b" + str(i + 1)] - self.optimizer.learning_rate * db_tmp
        return self.grads

    def _update_param(self):
        """
        Args
            AL: the output of the last layer
            Y: the answer vector of input data
        """

        self.optimizer.update(self.params, self.grads)


def relu(x: np.ndarray, max_val=None):
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


# def get_one_hot_value(x: np.ndarray) -> np.ndarray:
#     """one hot value for categorical. Max value -> 1, and the others -> 0."""

#     max_idx = np.argmax(x, axis=1)
#     res = np.zeros_like(x)
#     for i in range(len(max_idx)):
#         res[i, max_idx[i]] = 1
#     return res


def one_hot_encoding(x: pd.Series) -> np.ndarray:
    """convert categorical variable to binary vector."""

    unique_val = x.unique()
    unique_val.sort()
    return (x.values.reshape(-1, 1) == unique_val).astype(int)





# def GD_optimizer(w: np.ndarray, b: np.ndarray,
#                  X: np.ndarray, Y: np.ndarray,
#                  epoch: int = 1,
#                  lr: float = 0.001,
#                  print_cost: int = 0) -> tuple[np.ndarray, np.ndarray, list[float]]:
#     """
#     Args
#         w: weights (n_category, n_feature)
#         b: bias (n_category,)
#         X: train data X (n_data, n_feature)
#         Y: train data Y (n_data, n_category)
#         epoch: number of iterations
#         lr: learning rate
#         print_cost: printing the loss every given steps
    
#     Return
#         tuple(w, b, costs)
#         costs: cost list of every 100 steps
#     """

#     return BGD_optimizer(w, b, X, Y, epoch=epoch, batch=X.shape[0], lr=lr, print_cost=print_cost)


# def SGD_optimizer(w: np.ndarray, b: np.ndarray,
#                   X: np.ndarray, Y: np.ndarray,
#                   epoch: int = 1,
#                   lr: float = 0.001,
#                   print_cost: int = 0) -> tuple[np.ndarray, np.ndarray, list[float]]:
#     """
#     Args
#         w: weights (n_category, n_feature)
#         b: bias (n_category,)
#         X: train data X (n_data, n_feature)
#         Y: train data Y (n_data, n_category)
#         epoch: number of iterations
#         lr: learning rate
#         print_cost: printing the loss every given steps
    
#     Return
#         tuple(w, b, costs)
#         costs: cost list of every 100 steps
#     """

#     return BGD_optimizer(w, b, X, Y, epoch=epoch, batch=1, lr=lr, print_cost=print_cost)


# def BGD_optimizer(w: np.ndarray, b: np.ndarray,
#                   X: np.ndarray, Y: np.ndarray,
#                   epoch: int = 1,
#                   batch: int = 32,
#                   lr: float = 0.001,
#                   print_cost: int = 0) -> tuple[np.ndarray, np.ndarray, list[float]]:
#     """
#     Args
#         w: weights (n_category, n_feature)
#         b: bias (n_category,)
#         X: train data X (n_data, n_feature)
#         Y: train data Y (n_data, n_category)
#         epoch: number of iterations
#         batch: data size for updating w, b
#         lr: learning rate
#         print_cost: printing the loss every given steps
    
#     Return
#         tuple(w, b, costs)
#         costs: cost list of every 100 steps
#     """

#     w = copy.deepcopy(w)
#     b = copy.deepcopy(b)

#     costs = []

#     lim = len(X) // batch + (1 if len(X) % batch else 0)

#     for i in range(epoch):

#         # batch loop
#         for j in range(lim):
        
#             # calculate dw, db, cost
#             dw, db, cost = propagate(w, b, X[batch * j: batch * (j + 1)], Y[batch * j: batch * (j + 1)])

#             # update w, b
#             w = w - lr * dw
#             b = b - lr * db

#         if i % 100 == 0:
#             costs.append(cost)

#         if print_cost and i % print_cost == 0:
#             print (f"The cost of iteration {i}: {cost}")

#         # # check converged
#         # if (np.isclose(dw, np.zeros_like(dw), rtol=1e-6).all() or
#         #     np.isclose(db, np.zeros_like(db), rtol=1e-6).all()):
#         #     break

#     costs.append(cost)

#     return w, b, costs


# def predict(w: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
#     """
#     Args
#         w: weights (n_category, n_feature)
#         b: bias (n_category,)
#         X: train data X (n_data, n_feature)

#     Return
#         Y_pred: predict value (n_data, n_category)
#     """

#     m = X.shape[0]

#     A = sigmoid(X @ w.T + b)

#     if w.shape[0] == 1:
#         Y_pred = (A > 0.5).astype(int)
#         # for i in range(A.shape[1]):
#         #     Y_pred[i, 0] = 1 if A[i, 0] > 0.5 else 0
#     else:
#         Y_pred = get_one_hot_value(A)

#     return Y_pred.astype(int)


# def fit(x_train: np.ndarray, y_train: np.ndarray,
#           w_init: np.ndarray = None, b_init: np.ndarray = None,
#           epoch: int = 1,
#           batch: int = 32,
#           lr: int = 0.001,
#           print_cost: int = 100,
#           optimizer: str = "BGD") -> tuple[np.ndarray, np.ndarray, list[float]]:
#     """
#     Args
#         x_train: train data for input (n_data, n_feature)
#         y_train: train data for output (n_data, n_category)
#         w_init: init w (n_category, n_feature)
#         b_init: inti bias (n_category,)
#         epoch: number of iterations
#         batch: data size for updating w, b
#         lr: learning rate
#         print_cost: printing the cost every n steps. 0 or None for not printing.
    
#     Return
#         tuple(w, b, costs)
#         costs: cost list of every 100 steps
#     """

#     assert len(y_train.shape) in (1, 2), "invalid y value."
#     assert optimizer in ["BGD", "SGD", "GD"], "invalid optimizer."

#     if len(y_train.shape) == 1:
#         y_train = y_train.reshape(-1, 1)

#     n_feature = x_train.shape[1]
#     n_category = y_train.shape[1]

#     # w shape is (n_category, n_feature), and b shape is (n_category,)
#     if w_init is None:
#         w_init = np.zeros((n_category, n_feature))
#     if b_init is None:
#         b_init = np.zeros(n_category)

#     match optimizer:
#         case "BGD":
#             w, b, costs = BGD_optimizer(w_init, b_init, x_train, y_train, epoch, batch, lr, print_cost)
#         case "SGD":
#             w, b, costs = SGD_optimizer(w_init, b_init, x_train, y_train, epoch, lr, print_cost)
#         case "GD":
#             w, b, costs = GD_optimizer(w_init, b_init, x_train, y_train, epoch, lr, print_cost)

#     return w, b, costs



# def evaluate(w: np.ndarray, b: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
#     """return accuracy of prediction."""
    
#     y_pred = predict(w, b, x_test)
    
#     assert y_pred.shape == y_test.shape, "invalid input. different shape"

#     wrong = np.sum(np.abs(y_pred - y_test) / 2)
    
#     return (1 - wrong / len(y_test)) * 100



from load_csv import load

if __name__ == "__main__":

    data: pd.DataFrame = load("data.csv")

    assert data is not None, "data load failure."

    data_train = data
    # data_train = data[:450]
    # data_valid = data[450:]


    x_train = data_train.iloc[:, 2:].to_numpy()
    y_train = data_train.iloc[:, 1] == "M"
    m = data_train.iloc[:, 1] == "M"
    b = data_train.iloc[:, 1] == "B"
    y_train = pd.DataFrame({"M":m, "B":b})
    y_train = y_train.to_numpy().astype(int)

    mlp = Model.sequential([
        Layers.Input(x_train.shape[1]),
        Layers.Dense(24, activation='sigmoid', weights_initializer="heUniform"),
        Layers.Dense(24, activation='sigmoid', weights_initializer="heUniform"),
        Layers.Dense(24, activation='sigmoid', weights_initializer="heUniform"),
        Layers.Dense(y_train.shape[1], activation='softmax', weights_initializer="heUniform")
    ])

    mlp.compile(optimizer="adam", loss="binaryCrossentropy", metrics=['Accuracy'])

    history = mlp.fit(x_train, y_train, 1000, batch_size=32)