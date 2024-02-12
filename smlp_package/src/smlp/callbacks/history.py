from smlp.callbacks.callback import Callback


class History(Callback):
    """Model fitting history"""

    def __init__(self):
        super().__init__()
        self.history: dict[str, list] = {}

    def on_train_begin(self):
        self.epoch = []

    def on_epoch_end(self, epoch, logs: dict = None):
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
