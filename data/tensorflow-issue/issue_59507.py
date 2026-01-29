# tf.random.uniform((1,), dtype=tf.float32) ← input is a single vector of shape [1]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No additional layers needed as per original minimal example

    def call(self, inputs, training):
        # inputs expected to be a dict with key "x"
        # Just echo inputs with a print for tracing as in the original example
        # Here we simulate the print for demonstration (would be disabled under @tf.function)
        tf.print("Tracing with", inputs)
        return inputs

    def __call__(self, *args, **kwargs):
        # Explicitly call tf.keras.Model.__call__ to avoid the TypeError raised when saved as a signature
        return super().__call__(*args, **kwargs)

def my_model_function():
    model = MyModel()
    # Wrap the call method with tf.function assigned to __call__ to avoid the issue:
    # Instead of "model.__call__ = tf.function(model.__call__)" which fails,
    # use "model.__call__ = tf.function(model.call)" as suggested in the issue discussion.
    model.__call__ = tf.function(model.call)
    return model

def GetInput():
    # Return input as dict with key "x" and shape [1], dtype float32 as expected by concrete function signature
    return {"x": tf.random.uniform((1,), dtype=tf.float32)}

