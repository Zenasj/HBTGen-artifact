# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape is (batch_size, 10) based on test_model.test concrete function input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, dims=0, **kwargs):
        super().__init__(**kwargs)
        # Dense layer with dims outputs; dims may be zero (empty weights)
        self._dense = tf.keras.layers.Dense(dims)
    
    @tf.function
    def call(self, x):
        # Call the dense layer normally
        return self._dense(x)

def my_model_function():
    # Return an instance of MyModel with dims=0 to match the problematic case
    # This reproduces the scenario where the dense layer's weights are empty and cause TFLite conversion issues
    return MyModel(dims=0)

def GetInput():
    # Return a random tensor with shape (batch_size, 10), dtype float32
    # batch_size chosen as 1, consistent with the TFLite test code
    return tf.random.uniform((1, 10), dtype=tf.float32)

