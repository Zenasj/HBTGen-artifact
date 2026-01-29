# tf.random.uniform((B, None, 10), dtype=tf.float32)  # Assumed input shape: batch size unknown, time steps variable, features=10

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense layer with activity regularizer (L2)
        self.dense = tf.keras.layers.Dense(
            12,
            activity_regularizer=tf.keras.regularizers.l2(),
            name='dense'
        )
        # Lambda layer replicating identity, with activity regularizer L2
        # Note: Lambda layers have issues saving/loading activity_regularizer with h5 format.
        # We reproduce the original behavior.
        self.lambda_layer = tf.keras.layers.Lambda(
            lambda batch: batch,
            activity_regularizer=tf.keras.regularizers.l2(),
            name='lambda'
        )

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.lambda_layer(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns random input tensor of shape (batch_size, None, 10)
    # Since model input shape is (None, 10) with variable time steps,
    # choose a fixed sequence length for testing e.g., length=5, batch=4
    batch_size = 4
    seq_length = 5
    feature_dim = 10
    return tf.random.uniform(shape=(batch_size, seq_length, feature_dim), dtype=tf.float32)

