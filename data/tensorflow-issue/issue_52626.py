# tf.random.uniform((B, 100), dtype=tf.float32) ← input shape inferred from tf.keras.Input(shape=[100])

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a Dense layer that maps input shape (100,) to output shape (1,)
        self.dense = tf.keras.layers.Dense(1)
        # Concatenate layer which normally requires a list of tensors
        # Here it receives a single element list, replicating the no-op Concatenate layer scenario
        self.concatenate = tf.keras.layers.Concatenate()

    def call(self, inputs):
        # Apply Dense layer
        x = self.dense(inputs)
        # Wrap dense output in a list to feed Concatenate layer
        # This is the original pattern causing the reported loading problem
        x = self.concatenate([x])
        return x

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Return a random tensor that matches the expected input shape [batch_size, 100]
    # Batch size is chosen as 4 here arbitrarily
    batch_size = 4
    return tf.random.uniform((batch_size, 100), dtype=tf.float32)

