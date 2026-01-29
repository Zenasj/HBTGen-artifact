# tf.random.uniform((B, 3), dtype=tf.float32) ← Input shape inferred from placeholder (None, 3) in original code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Original `mlp` layer: Dense units=10
        self.mlp = tf.keras.layers.Dense(10)
        # Original `resize` layer: Dense units=2 (number of classes)
        self.resize = tf.keras.layers.Dense(2)
        # Dropout rate from Config
        self.drop_rate = 0.5

    def call(self, x, training=False):
        # Forward pass: mlp -> dropout -> resize
        z = self.mlp(x)
        # In TF2 Keras, dropout respects training flag
        z = tf.keras.layers.Dropout(rate=self.drop_rate)(z, training=training)
        z = self.resize(z)
        return z

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected shape (batch, 3)
    # Using batch size 1 for simplicity
    return tf.random.uniform((1, 3), dtype=tf.float32)

