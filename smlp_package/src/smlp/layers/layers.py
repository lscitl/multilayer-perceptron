from typing import Callable
from enum import Enum, auto
from smlp.initializers import Initializer
from smlp.initializers import glorotNormal
from smlp.initializers import glorotUniform
from smlp.initializers import heNormal
from smlp.initializers import heUniform, zeros
from smlp.initializers import zeros


class LAYER(Enum):
    INPUT = auto()
    DENSE = auto()
    # DROPOUT = auto()


class Layers:
    """
    layer class

    """

    layer_to_str = {
        LAYER.INPUT: "Input",
        LAYER.DENSE: "Dense",
        # LAYER.DROPOUT: "Dropout",
    }

    def __init__(
        self,
        layer_type: LAYER,
        layer_dim: int,
        activation: str,
        weights_initializer: str | Callable,
    ):
        """layer init."""
        self.type: LAYER = layer_type
        self.layer_dim = layer_dim
        self.activation = activation

        if layer_type != LAYER.INPUT:
            if isinstance(weights_initializer, str):
                match weights_initializer.lower():
                    case "henormal":
                        self.weights_initializer = heNormal()
                    case "heuniform":
                        self.weights_initializer = heUniform()
                    case "glorotnormal" | "xaviernormal":
                        self.weights_initializer = glorotNormal()
                    case "glorotuniform" | "xavieruniform":
                        self.weights_initializer = glorotUniform()
                    case "zero":
                        self.weights_initializer = zeros()
                    case _:
                        raise AssertionError("Invalid initializer.")

            elif isinstance(weights_initializer, Initializer):
                self.weights_initializer = weights_initializer

            else:
                raise AssertionError("Invalid initializer.")

    @staticmethod
    def Input(layer_dim: int):
        """Create dense layer."""

        return Layers(LAYER.INPUT, layer_dim, None, None)

    @staticmethod
    def Dense(
        layer_dim: int,
        activation: str | Callable,
        weights_initializer: str = "glorotNormal",
    ):
        """Create dense layer."""

        return Layers(LAYER.DENSE, layer_dim, activation, weights_initializer)

    @staticmethod
    def getLayer(layer: LAYER) -> str:
        return Layers.layer_to_str[layer]

    def getLayer(self) -> str:
        return self.layer_to_str[self.type]
