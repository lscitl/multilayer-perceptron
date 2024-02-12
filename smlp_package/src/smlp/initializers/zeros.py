import numpy as np
from smlp.initializers.initializer import Initializer


class zeros(Initializer):
    def __init__(self):
        super().__init__("zeros")

    def initialize(self, shape: tuple):
        """Zero initializer. For bias."""
        return np.zeros(shape)

    @staticmethod
    def initialize(shape: tuple):
        """Zero initializer. For bias."""
        return np.zeros(shape)
