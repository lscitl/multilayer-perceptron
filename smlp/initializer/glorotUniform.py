import numpy as np
from smlp.initializer.initializer import Initializer

class glorotUniform(Initializer):

    def __init__(self):
        super().__init__("glorotUniform")

    def initialize(self, shape: tuple):
        """Glorot / Xavier uniform initializer."""
        limit = np.sqrt(6.0 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, shape)
