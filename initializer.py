import numpy as np

class Initializer:
    """Initalizer"""

    @staticmethod
    def glorotNormal(shape: tuple):
        """Glorot / Xavier normal initializer."""
        stddev = np.sqrt(2.0 / (shape[0] + shape[1]))
        return np.random.normal(0, stddev, shape)

    @staticmethod
    def glorotUniform(shape: tuple):
        """Glorot / Xavier uniform initializer."""
        limit = np.sqrt(6.0 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    def heNormal(shape: tuple):
        """He normal initializer."""
        stddev = np.sqrt(2.0 / shape[0])
        return np.random.normal(0, stddev, shape)

    @staticmethod
    def heUniform(shape: tuple):
        """He uniform initializer."""
        limit = np.sqrt(6.0 / shape[0])
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    def zeros(shape: tuple):
        """Zero initializer. For bias."""
        return np.zeros(shape)

