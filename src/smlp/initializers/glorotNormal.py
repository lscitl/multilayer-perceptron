import numpy as np
from smlp.initializers.initializer import Initializer


class glorotNormal(Initializer):
    def __init__(self):
        super().__init__("glorotNormal")

    def initialize(self, shape: tuple):
        """Glorot / Xavier normal initializer."""
        stddev = np.sqrt(2.0 / (shape[0] + shape[1]))
        return np.random.normal(0, stddev, shape)
