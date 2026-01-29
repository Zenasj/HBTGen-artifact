# tf.random.normal((), dtype=tf.float32) ← scalar float tensor input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights or layers, model just returns a single random scalar tensor.
        # This mimics the TF graph mode behavior where tf.random.normal([]) returns a scalar tensor
        # producing fresh random values on each evaluation.
    
    def call(self, inputs=None):
        # Inputs are ignored; output a scalar random normal tensor.
        # This simulates the behavior from the reproduced issue where tf.random.normal([]) evaluated twice produces different values.
        return tf.random.normal(shape=[])

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # The model ignores input, but for compatibility, return None or a dummy tensor
    # Here we return a dummy tensor as model.call expects an input argument (or None)
    # Returning None is acceptable since call uses default inputs=None, else could be tf.zeros(()).
    return None

