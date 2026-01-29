# tf.random.uniform((1,), dtype=tf.float32) ← Input is a scalar variable wrapped in a module

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Emulate the original example where a variable is tracked inside a module
        self._var = tf.Variable(1.0, trainable=True, name='var')

    def call(self, inputs):
        # A simple forward pass that scales input by the tracked variable
        return inputs * self._var

def my_model_function():
    # Return an instance of MyModel with the variable initialized as in the original example
    return MyModel()

def GetInput():
    # Return a random scalar float32 tensor (batch size 1 for consistency)
    return tf.random.uniform(shape=(1,), dtype=tf.float32)

