from smlp.layers.layer import Layer, LAYER


class Input(Layer):
    """Create input layer."""

    def __init__(self, layer_dim: int):
        super().__init__(LAYER.INPUT, layer_dim, None, None)
