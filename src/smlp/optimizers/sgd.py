import numpy as np
from smlp.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        weight_decay=None,
        name="SGD",
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
                self.v["dW" + str(l)] = (
                    self.momentum * self.v["dW" + str(l)]
                    - self.lr * grads["dW" + str(l)]
                )
                self.v["db" + str(l)] = (
                    self.momentum * self.v["db" + str(l)]
                    - self.lr * grads["db" + str(l)]
                )

                if self.nesterov:
                    params["W" + str(l)] += (
                        self.momentum * self.v["dW" + str(l)]
                        - self.lr * grads["dW" + str(l)]
                    )
                    params["b" + str(l)] += (
                        self.momentum * self.v["db" + str(l)]
                        - self.lr * grads["db" + str(l)]
                    )
                else:
                    params["W" + str(l)] += self.v["dW" + str(l)]
                    params["b" + str(l)] += self.v["db" + str(l)]
