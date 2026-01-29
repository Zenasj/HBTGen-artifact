# tf.random.uniform((1, 10), dtype=tf.float32) ← Input shape as per Input(shape=(10,), batch_size=1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using a Sequential model inside this tf.keras.Model as per the original code
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(1024 * 5),
            tf.keras.layers.Reshape((5, 1, 1, 1024)),
            # BatchNormalization with axis=1 as in the original example
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.Dense(1024 * 5)
        ])

    def call(self, inputs, training=False):
        # Forward pass identical to original model
        return self.mlp(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Input shape matches batch_size=1 and input shape (10,)
    # Uniform random float32 tensor
    return tf.random.uniform((1, 10), dtype=tf.float32)

