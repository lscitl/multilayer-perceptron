import numpy as np
from smlp.optimizers.optimizer import Optimizer


class RMSprop(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-7,
        weight_decay=None,
        name="RMSprop",
    ):
        super().__init__(learning_rate, weight_decay, name)
        self.rho = rho
        self.momentum = momentum
        self.eps = epsilon

    def build(self, params):
        """Initialize adam params"""
        self.v, self.m = self._initialize_rmsprop(params)

    def _initialize_rmsprop(self, params: dict[str, np.ndarray]):
        """
        Args
            params: dictionary type of model params(weights, biases)

        Return
            v: the exponentially weighted average of the gradient
            m: update value(increment W, b) calculated by momentum
        """

        n_layer = len(params) // 2
        v = {}
        m = {}

        for l in range(1, n_layer + 1):
            dW = "dW" + str(l)
            W = "W" + str(l)
            db = "db" + str(l)
            b = "b" + str(l)

            v[dW] = np.zeros(params[W].shape)
            v[db] = np.zeros(params[b].shape)
            if self.momentum > 0:
                m[dW] = np.zeros(params[W].shape)
                m[db] = np.zeros(params[b].shape)

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
            dW = "dW" + str(l)
            W = "W" + str(l)
            db = "db" + str(l)
            b = "b" + str(l)

            self.v[dW] = rho * self.v[dW] + (1 - rho) * np.square(grads[dW])
            self.v[db] = rho * self.v[db] + (1 - rho) * np.square(grads[db])

            increment_W = np.divide(self.lr * grads[dW], np.sqrt(self.v[dW] + self.eps))
            increment_b = np.divide(self.lr * grads[db], np.sqrt(self.v[db] + self.eps))

            if self.momentum > 0:
                self.m[dW] = self.momentum * self.m[dW] + increment_W
                self.m[db] = self.momentum * self.m[db] + increment_b

                params[W] -= self.m[dW]
                params[b] -= self.m[db]

            else:
                params[W] -= increment_W
                params[b] -= increment_b
