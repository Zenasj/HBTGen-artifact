# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred from model Input(shape=(1,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A custom layer that returns a dictionary output with named tensors
        # This mimics the Layer class in the issue, which returns {"out1": ..., "out2": ...}
        self.my_layer = tf.keras.layers.Layer(name="my_layer")
        
    def call(self, inputs):
        # Instead of using a standard layer, implement call logic that returns dict with keys out1 and out2
        batch_size = tf.shape(inputs)[0]
        # Return two tensors of shape (batch_size,) as per original example (tf.repeat scalar 1 and 2)
        out1 = tf.repeat([1], batch_size)
        out2 = tf.repeat([2], batch_size)
        return {"out1": out1, "out2": out2}

def my_model_function():
    return MyModel()

def GetInput():
    # Return a batch of inputs consistent with Input(shape=(1,)) used in the example
    # Let's assume batch_size=8 as in the original data example (8 items)
    batch_size = 8
    # Inputs are shape (batch_size, 1), float32 (common default)
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)

