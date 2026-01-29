# tf.constant(42.) with shape () and dtype tf.float32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # No trainable variables as per original example

    def call(self, x):
        # Replicating the behavior of Sub.__call__ with a branch that uses super()
        # The original bug was triggered by using super() without args inside a conditional branch
        # The workaround was to use base_self = super(), then call base_self.__call__(x).
        base_self = super(MyModel, self)
        # Replicate the original conditional branch (True) that triggers the path.
        if True:
            # Call the Base.__call__ logic: in the original code it adds 1 to x
            # Since Base.__call__ was x + 1., we do the same:
            # Note: we call base_self.__call__(x) but base_self.__call__ is not defined here directly;
            # So instead, replicate Base behavior directly for clarity:
            return x + 1.
        else:
            return tf.constant(1., dtype=x.dtype)

def my_model_function():
    # Return instance of MyModel directly
    return MyModel()

def GetInput():
    # Return a scalar tensor matching the input expected by MyModel.call
    # Based on the original example, input was tf.constant(42.) with shape ()
    return tf.constant(42., dtype=tf.float32)

