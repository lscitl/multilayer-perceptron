#!/usr/bin/python3

import sys
import copy
import shutil
import time

import pandas as pd
import numpy as np

from typing import Any, Iterable
from functools import partial

from optimizer import Optimizer, Adam, SGD
from initializer import Initializer
from layers import LAYER, Layers
from callback import Callback, EarlyStopping, History


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

        self.history = None
        self.stop_training = False

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

    def compile(self, optimizer, loss, metrics=[]):
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
        
        self.optimizer.build(self.params)

        self.loss = loss
        self.metrics = metrics
        self.iscompile = True

    def fit(
        self,
        x: np.ndarray=None,
        y: np.ndarray=None,
        batch_size: int=32,
        epochs: int=1,
        callbacks: list[Callback]=None,
        validation_split: float=0.0
    ):
        """train data."""

        self._assert_compile()

        self.stop_training = False
        self.history = History()
        self.history.set_model(self)

        callback_tmp = []
        if callbacks:
            callback_tmp = callbacks[:]
        callback_tmp.append(self.history)

        self._callback_on_train(callback_tmp)

        batch_max, batch_remain = divmod(x.shape[0], batch_size)
        if batch_remain:
            batch_max += 1
        # seed = int(time.time() * 1000000)
        seed = 10

        for i in range(epochs):
            loss_sum = 0
            loss = 0
            acc_sum = 0
            acc = 0
            train_size = 0
            max_str_len = 0
            logs = {}

            # for mini-batch
            np.random.seed(seed)
            permutation = list(np.random.permutation(x.shape[0]))
            shuffled_x = x[permutation, :]
            shuffled_y = y[permutation, :]

            epoch_str = f"{i + 1}/{epochs}"
            print(epoch_str)

            for n_batch in range(batch_max):

                batch_start_time = time.time()
                mini_batch_x = shuffled_x[n_batch * batch_size: (n_batch + 1) * batch_size]
                mini_batch_y = shuffled_y[n_batch * batch_size: (n_batch + 1) * batch_size]

                train_size += mini_batch_x.shape[0]
                AL, caches = self._model_forward(mini_batch_x)
                loss_sum += self._compute_cost(AL, mini_batch_y) * mini_batch_x.shape[0]
                loss = loss_sum / train_size

                self._model_backward_and_update_params(AL, mini_batch_y, caches)

                batch_end_time = time.time()
                batch_time_micro_s = int((batch_end_time - batch_start_time) * 1000000)
                batch_time_ms, batch_time_micro_remain = divmod(batch_time_micro_s, 1000)
                
                batch_str = f"\r{n_batch + 1}/{batch_max}"

                batch_time_s = 0
                if batch_time_ms == 0:
                    batch_str += f" - 0s {batch_time_micro_remain}μs/step"
                else:
                    if batch_time_ms > 1000:
                        batch_time_s, batch_time_ms = divmod(batch_time_ms, 1000)
                    batch_str += f" - {batch_time_s}s {batch_time_ms}ms/step"
                
                batch_str += f" - loss: {loss:.4f}"

                if hasattr(self.metrics, "Accuracy"):
                    acc_tmp = self._get_accuracy_sum(AL, mini_batch_y)
                    if acc_tmp is not None:
                        acc_sum += acc_tmp
                        acc = acc_sum / train_size
                        batch_str += f" - accuracy: {acc:.4f}"
                
                if max_str_len < len(batch_str):
                    max_str_len = len(batch_str)
                else:
                    batch_str += " " * (max_str_len - len(batch_str))
                print(batch_str, end="", flush=True)

            seed += 1
            print("")

            # for callback in callbacks:
            #     callback

            if self.stop_training:
                break

        return history

    def _callback_on_train(self, callback: list[Callback]):
        for cb in callback:
            cb.on_train_begin()

    def _get_accuracy_sum(self, y_hat: np.ndarray, y: np.ndarray):
        """get accuracy"""

        match self.loss:
            case "binaryCrossentropy":
                if y_hat.shape[1] != 1:
                    y_hat = get_one_hot_value(y_hat)
                    return np.sum(np.all(y_hat == y, axis=1))
                elif y.ndim == 1 or y.shape[1] == 1:
                    y_hat = y_hat >= 0.5
                    return np.sum(y_hat)
                else:
                    raise AssertionError("loss error")
            case _:
                return None


    def evaluate(self,
                 x=None,
                 y=None):
        """evaluate the model."""

        self._assert_compile()

    def predict(self,
                x=None):
        """predict."""

        self._assert_compile()

        predict, _ = self._model_forward(x)

        match self.loss:
            case "binaryCrossentropy":
                if predict.shape[1] != 1:
                    return get_one_hot_value(predict)

        return predict

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

        self.optimizer.update(self.params, self.grads)

        return self.grads


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

    history = mlp.fit(x_train, y_train, batch_size=32, epochs=1000)