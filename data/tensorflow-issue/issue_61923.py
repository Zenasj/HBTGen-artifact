# tf.random.uniform((B, 10), dtype=tf.float32)  ← Assumed batch size B, input shape (10,) based on input_dim=10

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Encoder layers
        self.encoder_input = tf.keras.layers.InputLayer(input_shape=(10,), name="all_inputs")
        self.latent_layer = tf.keras.layers.Dense(5, activation="relu", name="latent_space")

        # Decoder layers
        self.decoder_input = tf.keras.layers.InputLayer(input_shape=(5,), name="latent_input")
        self.decoder_hidden = tf.keras.layers.Dense(10, activation="relu", name="decoder2")
        self.numeric_output_layer = tf.keras.layers.Dense(3, activation="linear", name="numeric_output")
        self.binary_output_layer = tf.keras.layers.Dense(2, activation="sigmoid", name="binary_output")

    def call(self, inputs, training=False):
        # Forward pass through encoder
        x = self.encoder_input(inputs)
        latent = self.latent_layer(x)

        # Forward pass through decoder
        dec_in = self.decoder_input(latent)
        dec_hidden = self.decoder_hidden(dec_in)
        numeric_output = self.numeric_output_layer(dec_hidden)
        binary_output = self.binary_output_layer(dec_hidden)

        # Return list outputs to match the original decoder outputs
        return [numeric_output, binary_output]


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Input shape: (batch_size, 10), dtype float32 as typical for numeric data
    batch_size = 32  # typical batch size, can be changed
    return tf.random.uniform((batch_size, 10), dtype=tf.float32)

