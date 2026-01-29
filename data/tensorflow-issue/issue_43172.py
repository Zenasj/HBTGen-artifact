# tf.random.uniform((B, 2), dtype=tf.float16) ← Input shape inferred from Dense input_dim=2 in example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer with 1 unit and float16 dtype, matching example model
        self.dense = tf.keras.layers.Dense(units=1, dtype=tf.float16, input_shape=(2,))

    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    # Create an instance of MyModel and initialize weights similarly to example
    model = MyModel()

    # Initialize with specific weights to mimic the example where NaNs could arise
    # w and b are float16 numpy arrays as in the chunk
    import numpy as np
    w = np.array([[-0.2373], [1.0]], dtype=np.float16)  # Weight matrix shape (2,1)
    b = np.array([0.0], dtype=np.float16)              # Bias vector shape (1,)
    model.dense.build((None, 2))  # build model weights with input shape (None, 2)
    model.dense.set_weights([w, b])

    return model

def GetInput():
    # Return a random float16 tensor with shape (batch_size=1, input_dim=2)
    # consistent with batch_size=1 causing issues in the original example
    return tf.random.uniform((1, 2), minval=0.0, maxval=1.0, dtype=tf.float16)

