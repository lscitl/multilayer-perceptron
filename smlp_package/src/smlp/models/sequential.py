from typing import Iterable
from smlp.models.model import Model
from smlp.layers.layer import Layer, LAYER


class Sequential(Model):
    """Sequential class for multilayer perceptron model."""

    def __init__(self, layers: Iterable[Layer]):
        """Create modle with given layer"""

        super().__init__()

        for layer in layers:
            assert isinstance(layer, Layer), "invalid layer is included."
            self.add(layer)

    def add(self, layer: Layer):
        """add layer"""

        assert isinstance(layer, Layer), "invalid layer type."

        if len(self.layer) == 0:
            assert layer.type == LAYER.INPUT, "first layer should be INPUT layer."
            self.layer.append(layer)

        else:
            assert (
                layer.type != LAYER.INPUT
            ), "Input layer can be set only in the first layer."
            self.layer.append(layer)
