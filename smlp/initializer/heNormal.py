import numpy as np
from smlp.initializer.initializer import Initializer

class heNormal(Initializer):

    def __init__(self):
        super().__init__("heNormal")

    def initialize(self, shape: tuple):
        """He normal initializer."""
        stddev = np.sqrt(2.0 / shape[0])
        return np.random.normal(0, stddev, shape)
