# This code adapts Improved versions of ModelCheckpoint and EarlyStopping callbacks 
# with properly working set_params and get_params methods as described in the issue.

import tensorflow as tf
import warnings

class MyModel(tf.keras.Model):
    """
    This class fuses updated versions of ModelCheckpoint and EarlyStopping callbacks
    that have properly functioning set_params and get_params methods. 
    
    Since the original issue is about callbacks parameter mutation/fixing, 
    this "model" class is a wrapper encapsulating both updated callbacks as submodules,
    demonstrating the fix in set_params behavior.
    """

    def __init__(self, ckpt_path='/content/'):
        super().__init__()
        # Instantiate updated callbacks as submodules
        self.checkpoint = UpdatedModelCheckpoint(ckpt_path)
        self.early_stopping = UpdatedEarlyStopping()
    
    def call(self, inputs, training=None):
        """
        Since callbacks are not typical model layers, the forward step here is a no-op.
        But to fulfill the interface, we return a dict of current parameters for both 
        submodules to demonstrate the parameter setting and getting functionality.
        """
        return {
            'checkpoint_params': self.checkpoint.get_params(),
            'early_stopping_params': self.early_stopping.get_params()
        }

class UpdatedModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """
    Updated version of ModelCheckpoint with a working set_params and get_params method.
    """

    def set_params(self, params: dict) -> None:
        """
        Allow setting of the callback's parameters from a dictionary.
        Warn on invalid keys.
        """
        if not isinstance(params, dict):
            raise ValueError('`params` must be a dictionary mapping parameter names to values.')
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                warnings.warn(f'Invalid parameter -> {param} !', category=UserWarning)

    def get_params(self) -> dict:
        """
        Return a shallow copy of the callback's __dict__ (all attributes).
        """
        return self.__dict__.copy()

class UpdatedEarlyStopping(tf.keras.callbacks.EarlyStopping):
    """
    Updated version of EarlyStopping with a working set_params and get_params method.
    """

    def set_params(self, params: dict) -> None:
        """
        Allow setting of the callback's parameters from a dictionary.
        Warn on invalid keys.
        """
        if not isinstance(params, dict):
            raise ValueError('`params` must be a dictionary mapping parameter names to values.')
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                warnings.warn(f'Invalid parameter -> {param} !', category=UserWarning)

    def get_params(self) -> dict:
        """
        Return a shallow copy of the callback's __dict__ (all attributes).
        """
        return self.__dict__.copy()

def my_model_function():
    """
    Returns an instance of MyModel with default checkpoint path '/content/'.
    This model contains the fixed callbacks inside.
    """
    return MyModel()

def GetInput():
    """
    Return a dummy tensor input compatible with MyModel call interface.
    Since MyModel.call expects an input tensor but does no computation,
    we return a randomly generated tensor of shape (1, 10) float32.
    This is arbitrary since the model logic does not depend on input data.
    """
    return tf.random.uniform((1, 10), dtype=tf.float32)

