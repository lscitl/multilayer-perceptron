
class Callback:
    """callbacks"""

    def __init__(self):
        self.model = None

    def set_model(self, model):
        """set model for changing model params"""
        self.model = model
    
    def on_epoch_end(self, epoch, logs: dict=None):
        """function for end of epoch"""
        pass

    def on_train_begin(self):
        """function for beginning of train"""
        pass

    def on_train_end(self, logs=None):
        """function for end of train"""
        pass

class EarlyStopping(Callback):
    """Early stopping"""

    def __init__(self, monitor = "val_loss", patience=0, mode="auto", start_from_epoch=0):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.start_from_epoch = start_from_epoch
        self.patience_count = 0
        self.best = None
        self.monitor_op = None
        
        match mode:
            case "max":
                self.monitor_op = eq_greater
            case "min":
                self.monitor_op = eq_less
            case _:
                if "loss" in monitor:
                    self.monitor_op = eq_less
                elif "acc" in monitor:
                    self.monitor_op = eq_greater
                else:
                    AssertionError("Mode 'auto' is not supported for current monitor value.")

    def on_epoch_end(self, epoch, logs: dict=None):

        current = logs.get(self.monitor)
        if current is None:
            ValueError("monitor value is not valid.")

        if epoch >= self.start_from_epoch:
            if self.best is None:
                self.best = logs.get(self.monitor)
                if self.best is None:
                    print(f"Early stopping is not working for {self.monitor}")
            else:
                # if monitor_op value is true, it is early stop condition.
                if self.monitor_op(self.best, current):
                    self.patience_count += 1
                else:
                    self.best = current
                    self.patience_count = 0
                
                if self.patience_count > self.patience:
                    self.model.stop_training = True

    def on_train_end(self, logs=None):
        pass

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

def eq_greater(a, b):
    return a >= b

def eq_less(a, b):
    return a <= b