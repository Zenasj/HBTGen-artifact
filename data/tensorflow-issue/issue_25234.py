# tf.random.uniform((B, 3), dtype=tf.float32) ← Based on input X_train shape (100, 3) used in the issue reproduction

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original model was a simple Sequential with one Dense(1) layer.
        # We'll replicate that here.
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Return a fresh instance of the simple model used in the issue.
    # This matches the model trained on inputs with shape (None, 3) and output (None, 1).
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected input shape (batch_size, 3).
    # The batch size is arbitrary here; let's choose 8 for demonstration.
    return tf.random.uniform((8, 3), dtype=tf.float32)

