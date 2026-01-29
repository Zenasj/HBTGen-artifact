# tf.random.uniform((B, 40, 4), dtype=tf.float32) ← inferred input shape: (batch_size, window_length=40, feats=4)

import tensorflow as tf
from tensorflow import keras

window_length = 40
feats = 4

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers as in the original Sequential model
        # Batch input shape is (None, 40, 4)
        # Use kernel_initializer as 'he_uniform', return_sequences configured per layer

        self.encoder_1 = keras.layers.LSTM(
            64, kernel_initializer='he_uniform',
            return_sequences=True, name='encoder_1')
        self.dropout_1 = keras.layers.Dropout(0.25)

        self.encoder_2 = keras.layers.LSTM(
            32, kernel_initializer='he_uniform',
            return_sequences=True, name='encoder_2')
        self.dropout_2 = keras.layers.Dropout(0.25)

        self.encoder_3 = keras.layers.LSTM(
            16, kernel_initializer='he_uniform',
            return_sequences=False, name='encoder_3')
        self.dropout_3 = keras.layers.Dropout(0.25)

        self.encoder_decoder_bridge = keras.layers.RepeatVector(window_length, name='encoder_decoder_bridge')

        self.decoder_1 = keras.layers.LSTM(
            16, kernel_initializer='he_uniform',
            return_sequences=True, name='decoder_1')
        self.dropout_4 = keras.layers.Dropout(0.25)

        self.decoder_2 = keras.layers.LSTM(
            32, kernel_initializer='he_uniform',
            return_sequences=True, name='decoder_2')
        self.dropout_5 = keras.layers.Dropout(0.25)

        self.decoder_3 = keras.layers.LSTM(
            64, kernel_initializer='he_uniform',
            return_sequences=True, name='decoder_3')
        self.dropout_6 = keras.layers.Dropout(0.25)

        self.output_layer = keras.layers.TimeDistributed(keras.layers.Dense(feats))

    def call(self, inputs, training=False):
        """
        Forward pass mimics the original Sequential model architecture.
        Inputs shape: (batch_size, 40, 4)
        """
        x = self.encoder_1(inputs, training=training)
        x = self.dropout_1(x, training=training)

        x = self.encoder_2(x, training=training)
        x = self.dropout_2(x, training=training)

        x = self.encoder_3(x, training=training)
        x = self.dropout_3(x, training=training)

        x = self.encoder_decoder_bridge(x)

        x = self.decoder_1(x, training=training)
        x = self.dropout_4(x, training=training)

        x = self.decoder_2(x, training=training)
        x = self.dropout_5(x, training=training)

        x = self.decoder_3(x, training=training)
        x = self.dropout_6(x, training=training)

        output = self.output_layer(x)
        return output

def my_model_function():
    """
    Instantiates MyModel and compiles it similarly to the original code.
    """
    model = MyModel()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00005), loss='mse')
    return model

def GetInput():
    """
    Returns a random tensor input matching the expected input shape:
    (batch_size, window_length=40, feats=4).
    Here batch size is chosen arbitrarily as 32 for demonstration.
    """
    batch_size = 32
    return tf.random.uniform((batch_size, window_length, feats), dtype=tf.float32)

