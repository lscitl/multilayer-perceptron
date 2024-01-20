
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
