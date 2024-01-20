import numpy as np
from smlp.optimizer.optimizer import Optimizer

class RMSprop(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-7,
        weight_decay=None,
        name="RMSprop"
    ):
        super().__init__(learning_rate, weight_decay, name)
        self.rho = rho
        self.momentum = momentum
        self.eps = epsilon

    def build(self, params):
        """Initialize adam params"""
        self.v, self.m = self._initialize_rmsprop(params)

    def _initialize_rmsprop(self, params: dict[str, np.ndarray]) :
        """
        Args
            params: dictionary type of model params(weights, biases)
        
        Return
            v: the exponentially weighted average of the gradient
            s: the exponentially weighted average of the squared gradient
        """
        
        n_layer = len(params) // 2
        v = {}
        m = {}
        
        for l in range(1, n_layer + 1):
            v["dW" + str(l)] = np.zeros(params["W" + str(l)].shape)
            v["db" + str(l)] = np.zeros(params["b" + str(l)].shape)
            if self.momentum > 0:
                m["dW" + str(l)] = np.zeros(params["W" + str(l)].shape)
                m["db" + str(l)] = np.zeros(params["b" + str(l)].shape)
        
        return v, m

    def update(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]):
        """
        Args
            params: dictonary type of weights and biases
            grads: dictonary type of gradients for each params
        """

        n_layer = len(params) // 2
        rho = self.rho

        for l in range(1, n_layer + 1):
            self.v["dW" + str(l)] = rho * self.v["dW" + str(l)] + (1 - rho) * np.square(grads["dW" + str(l)])
            self.v["db" + str(l)] = rho * self.v["db" + str(l)] + (1 - rho) * np.square(grads["db" + str(l)])

            increment_W = np.divide(self.lr * grads["dW" + str(l)], np.sqrt(self.v["dW" + str(l)] + self.eps))
            increment_b = np.divide(self.lr * grads["db" + str(l)], np.sqrt(self.v["db" + str(l)] + self.eps))

            if self.momentum > 0:
                self.m["dW" + str(l)] = self.momentum * self.m["dW" + str(l)] + increment_W
                self.m["db" + str(l)] = self.momentum * self.m["db" + str(l)] + increment_b

                params["W" + str(l)] -= self.m["dW" + str(l)]
                params["b" + str(l)] -= self.m["db" + str(l)]

            else:
                params["W" + str(l)] -= increment_W
                params["b" + str(l)] -= increment_b

