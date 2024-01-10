#!/usr/bin/python3

import numpy as np

class Optimizer:
    def __init__(self, learning_rate=None, weight_decay=None, name=None):
        self.learning_rate = learning_rate
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

    def build(self, params):
        """Initialize adam params"""

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


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        weight_decay=None,
        name="adam"
    ):
        super().__init__(learning_rate, weight_decay, name)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, params):
        """Initialize adam params"""
        self.v, self.s = self._initialize_adam(params)
        self.t = 0

    def _initialize_adam(params: dict[str, np.ndarray]) :
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
        """update weight"""

        self.t += 1
        self._update_parameters_with_adam(params, grads)

    def _update_parameters_with_adam(
            self,
            params: dict[str, np.ndarray],
            grads: dict[str, np.ndarray]
        ):
        """
        Args
            params: dictonary type of weights and biases
            grads: dictonary type of gradients for each params
        """
        
        n_layer = len(params) // 2
        v_corrected = {}
        s_corrected = {}
        
        for l in range(1, n_layer + 1):
            self.v["dW" + str(l)] = self.beta_1 * self.v["dW" + str(l)] + (1 - self.beta_1) * grads["dW" + str(l)]
            self.v["db" + str(l)] = self.beta_1 * self.v["db" + str(l)] + (1 - self.beta_1) * grads["db" + str(l)]

            v_corrected["dW" + str(l)] = self.v["dW" + str(l)] / (1 - self.beta_1 ** self.t)
            v_corrected["db" + str(l)] = self.v["db" + str(l)] / (1 - self.beta_1 ** self.t)

            self.s["dW" + str(l)] = self.beta_2 * self.s["dW" + str(l)] + (1 - self.beta_2) * grads["dW" + str(l)] * grads["dW" + str(l)]
            self.s["db" + str(l)] = self.beta_2 * self.s["db" + str(l)] + (1 - self.beta_2) * grads["db" + str(l)] * grads["db" + str(l)]

            s_corrected["dW" + str(l)] = self.s["dW" + str(l)] / (1 - self.beta_2 ** self.t)
            s_corrected["db" + str(l)] = self.s["db" + str(l)] / (1 - self.beta_2 ** self.t)

            params["W" + str(l)] -= self.learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + self.epsilon)
            params["b" + str(l)] -= self.learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + self.epsilon)
