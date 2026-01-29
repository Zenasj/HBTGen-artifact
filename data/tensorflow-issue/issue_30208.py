# tf.random.uniform((B, 4), dtype=tf.float32)  ← Inferred input shape from original model input_shape=(4,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the original minimal example from the issue:
        # Sequential model with two Dense layers:
        #  - Dense(5, input_shape=(4,), activation='sigmoid')
        #  - Dense(3, input_shape=(5,), use_bias=True)
        # Here, the second layer's input_shape argument is redundant in Sequential API
        # and will be inferred automatically, but we keep the logic inline.
        self.dense1 = tf.keras.layers.Dense(5, activation='sigmoid', input_shape=(4,))
        self.dense2 = tf.keras.layers.Dense(3, use_bias=True)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # From the original issue, input_shape=(4,) for the first layer
    # Use batch size 2 as a reasonable default
    batch_size = 2  # reasonable default batch size for test
    return tf.random.uniform((batch_size, 4), dtype=tf.float32)

