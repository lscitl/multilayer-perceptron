#!/usr/bin/python3

import numpy as np

class Optimizer:
    def __init__(self, learning_rate=None, weight_decay=None, name=None):
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.name = name

    def build(self):
        pass

    def update(self, params, grads):
        pass


class SGD(Optimizer):
    def __init__(
        self,
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        weight_decay=None,
        name="SGD"
    ):
        super().__init__(learning_rate, weight_decay, name)

        if not isinstance(momentum, float) or momentum < 0 or momentum > 1:
            raise ValueError("`momentum` must be a float between [0, 1].")

        self.momentum = momentum
        self.nesterov = nesterov
        if self.nesterov and self.momentum == 0:
            self.nesterov = False

    def build(self, params):
        """Initialize adam params"""

        self.v = {}
        if self.momentum != 0:
            self.v = self._initialize_velocity(params)

    def _initialize_velocity(self, params: dict[str, np.ndarray]):
        """
        Args
            params: dictionary type of model params(weights, biases)
        
        Return
            v: current velocity
        """
        
        n_layer = len(params) // 2
        v = {}
        
        for l in range(1, n_layer + 1):
            v["dW" + str(l)] = np.zeros(params["W" + str(l)].shape)
            v["db" + str(l)] = np.zeros(params["b" + str(l)].shape)
            
        return v

    def update(self, params, grads):
        """update weight"""

        n_layer = len(params) // 2

        for l in range(1, n_layer + 1):
            if len(self.v) == 0:
                params["W" + str(l)] -= self.lr * grads["dW" + str(l)]
                params["b" + str(l)] -= self.lr * grads["db" + str(l)]

            else:
                self.v["dW" + str(l)] = self.momentum * self.v["dW" + str(l)] - self.lr * grads["dW" + str(l)]
                self.v["db" + str(l)] = self.momentum * self.v["db" + str(l)] - self.lr * grads["db" + str(l)]
                
                if self.nesterov:
                    params["W" + str(l)] += self.momentum * self.v["dW" + str(l)] - self.lr * grads["dW" + str(l)]
                    params["b" + str(l)] += self.momentum * self.v["db" + str(l)] - self.lr * grads["db" + str(l)]
                else:
                    params["W" + str(l)] += self.v["dW" + str(l)]
                    params["b" + str(l)] += self.v["db" + str(l)]


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        weight_decay=None,
        name="adam"
    ):
        super().__init__(learning_rate, weight_decay, name)
        self.b1 = beta_1
        self.b2 = beta_2
        self.eps = epsilon

    def build(self, params):
        """Initialize adam params"""
        self.v, self.s = self._initialize_adam(params)
        self.t = 0

    def _initialize_adam(self, params: dict[str, np.ndarray]) :
        """
        Args
            params: dictionary type of model params(weights, biases)
        
        Return
            v: the exponentially weighted average of the gradient
            s: the exponentially weighted average of the squared gradient
        """
        
        n_layer = len(params) // 2
        v = {}
        s = {}
        
        for l in range(1, n_layer + 1):
            v["dW" + str(l)] = np.zeros(params["W" + str(l)].shape)
            v["db" + str(l)] = np.zeros(params["b" + str(l)].shape)
            s["dW" + str(l)] = np.zeros(params["W" + str(l)].shape)
            s["db" + str(l)] = np.zeros(params["b" + str(l)].shape)
        
        return v, s

    def update(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]):
        """
        Args
            params: dictonary type of weights and biases
            grads: dictonary type of gradients for each params
        """

        self.t += 1

        n_layer = len(params) // 2

        beta_1_power = self.b1 ** self.t
        beta_2_power = self.b2 ** self.t
        
        for l in range(1, n_layer + 1):
            self.v["dW" + str(l)] = self.b1 * self.v["dW" + str(l)] + (1 - self.b1) * grads["dW" + str(l)]
            self.v["db" + str(l)] = self.b1 * self.v["db" + str(l)] + (1 - self.b1) * grads["db" + str(l)]

            v_corrected_W = self.v["dW" + str(l)] / (1 - beta_1_power)
            v_corrected_b = self.v["db" + str(l)] / (1 - beta_1_power)

            self.s["dW" + str(l)] = self.b2 * self.s["dW" + str(l)] + (1 - self.b2) * np.square(grads["dW" + str(l)])
            self.s["db" + str(l)] = self.b2 * self.s["db" + str(l)] + (1 - self.b2) * np.square(grads["db" + str(l)])

            s_corrected_W = self.s["dW" + str(l)] / (1 - beta_2_power)
            s_corrected_b = self.s["db" + str(l)] / (1 - beta_2_power)

            params["W" + str(l)] -= self.lr * v_corrected_W / (np.sqrt(s_corrected_W) + self.eps)
            params["b" + str(l)] -= self.lr * v_corrected_b / (np.sqrt(s_corrected_b) + self.eps)

