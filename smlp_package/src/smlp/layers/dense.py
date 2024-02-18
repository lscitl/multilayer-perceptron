from typing import Callable
from smlp.layers.layer import Layer, LAYER


class Dense(Layer):
    """Create dense layer."""

    def __init__(
        self,
        layer_dim: int,
        activation: str | Callable,
        weights_initializer: str = "glorotNormal",
    ):
        super().__init__(LAYER.DENSE, layer_dim, activation, weights_initializer)
