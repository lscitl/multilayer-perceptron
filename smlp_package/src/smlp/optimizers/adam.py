import numpy as np
from smlp.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        weight_decay=None,
        name="adam",
    ):
        super().__init__(learning_rate, weight_decay, name)
        self.b1 = beta_1
        self.b2 = beta_2
        self.eps = epsilon

    def build(self, params):
        """Initialize adam params"""
        self.v, self.s = self._initialize_adam(params)
        self.t = 0

    def _initialize_adam(self, params: dict[str, np.ndarray]):
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

        beta_1_power = self.b1**self.t
        beta_2_power = self.b2**self.t

        for l in range(1, n_layer + 1):
            dW = "dW" + str(l)
            W = "W" + str(l)
            db = "db" + str(l)
            b = "b" + str(l)

            self.v[dW] = self.b1 * self.v[dW] + (1 - self.b1) * grads[dW]
            self.v[db] = self.b1 * self.v[db] + (1 - self.b1) * grads[db]

            v_corrected_W = self.v[dW] / (1 - beta_1_power)
            v_corrected_b = self.v[db] / (1 - beta_1_power)

            self.s[dW] = self.b2 * self.s[dW] + (1 - self.b2) * np.square(grads[dW])
            self.s[db] = self.b2 * self.s[db] + (1 - self.b2) * np.square(grads[db])

            s_corrected_W = self.s[dW] / (1 - beta_2_power)
            s_corrected_b = self.s[db] / (1 - beta_2_power)

            params[W] -= self.lr * v_corrected_W / (np.sqrt(s_corrected_W) + self.eps)
            params[b] -= self.lr * v_corrected_b / (np.sqrt(s_corrected_b) + self.eps)
