class Optimizer:
    def __init__(self, learning_rate=None, weight_decay=None, name=None):
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.name = name

    def build(self):
        pass

    def update(self, params, grads):
        pass
