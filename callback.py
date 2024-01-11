
class Callback:
    """callbacks"""

    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_train_begin(self):
        pass
    
    def on_epoch_end(self, epoch, logs: dict=None):
        pass

class EarlyStopping(Callback):
    """Early stopping"""

    def __init__(self, monitor = "val_loss", patience=0):
        super().__init__()
        self.monitor = monitor
        self.patience = patience

    def on_epoch_end(self, epoch, logs: dict=None):
        return


class History(Callback):
    """Model fitting history"""

    def __init__(self):
        super().__init__()
        self.history: dict[str, list] = {}

    def on_train_begin(self):
        self.epoch = []

    def on_epoch_end(self, epoch, logs: dict=None):
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
