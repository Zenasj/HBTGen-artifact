# tf.random.uniform((2, 3, 2), dtype=tf.float32) ← Input shape inferred from example: batch=2, timesteps=3, features=2

import tensorflow as tf

class EncodingLayer(tf.keras.layers.Layer):
    def __init__(self, out_size):
        super().__init__()
        # Use GRU layer with return_sequences=True and return_state=True
        self.rnn_layer = tf.keras.layers.GRU(
            out_size,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def call(self, X, **kwargs):
        # Forward pass returns both output sequences and last state
        output, state = self.rnn_layer(X)
        return output, state

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Compose the encoding layer as a sub-layer
        self.encoder_layer = EncodingLayer(out_size=1)

    def call(self, inputs, **kwargs):
        # Using call instead of a separate method like infer
        output, state = self.encoder_layer(inputs)
        return output

def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random float32 tensor matching (2, 3, 2)
    # Same shape as used in the original repro example
    return tf.random.uniform(shape=(2, 3, 2), dtype=tf.float32)

