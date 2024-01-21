import numpy as np
from smlp.initializers.initializer import Initializer


class heUniform(Initializer):
    def __init__(self):
        super().__init__("heUniform")

    def initialize(self, shape: tuple):
        """He uniform initializer."""
        limit = np.sqrt(6.0 / shape[0])
        return np.random.uniform(-limit, limit, shape)
