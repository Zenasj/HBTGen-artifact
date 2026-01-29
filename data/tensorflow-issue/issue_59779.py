# tf.random.uniform((B, None, 5), dtype=tf.float32) ← inferred input shape from keras.layers.Input(shape=(None, dim), dim=5)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l1, l2, l1_l2

class MyModel(tf.keras.Model):
    def __init__(self, layer_size=128, dim=5, dropout=0.2):
        super().__init__()
        # Preprocessing: Gaussian Noise after Input
        # Note: ragged=True was specified; here we assume dense input for generality.
        self.pre = keras.Sequential([
            keras.layers.Input(shape=(None, dim)),
            keras.layers.GaussianNoise(0.2)
        ])

        # Three stacked LSTMs with regularization, batch norm, and residual connection
        self.rnn1 = keras.layers.LSTM(layer_size, 
                                     return_sequences=True, return_state=True, 
                                     dropout=dropout, 
                                     kernel_regularizer=l1(1e-6), 
                                     recurrent_regularizer=l1(1e-6), 
                                     bias_regularizer=l1(1e-6))
        self.bn = keras.layers.BatchNormalization()
        self.rnn2 = keras.layers.LSTM(layer_size, 
                                     return_sequences=True, return_state=True,
                                     dropout=dropout,
                                     kernel_regularizer=l2(1e-6), 
                                     recurrent_regularizer=l2(1e-6), 
                                     bias_regularizer=l2(1e-6))
        self.res = keras.layers.Add()
        self.rnn3 = keras.layers.LSTM(layer_size,
                                     return_sequences=False, return_state=True,
                                     dropout=dropout,
                                     kernel_regularizer=l1_l2(1e-6),
                                     recurrent_regularizer=l1_l2(1e-6),
                                     bias_regularizer=l1_l2(1e-6))

        # Dense layers for first output branch with batch normalization
        self.dense1 = keras.layers.Dense(32, 
                                         kernel_regularizer=l1(1e-5),
                                         bias_regularizer=l2(1e-5))
        self.bn2 = keras.layers.BatchNormalization()
        self.dense2 = keras.layers.Dense(16,
                                         kernel_regularizer=l2(1e-5),
                                         bias_regularizer=l1_l2(1e-5))
        self.hosp_out = keras.layers.Dense(1, activation="sigmoid")

        # Dense layers for second output branch, including Dropout
        self.dense4 = keras.layers.Dense(128,
                                         kernel_regularizer=l1(1e-5),
                                         bias_regularizer=l2(1e-5))
        self.dropout3 = keras.layers.Dropout(0.2)
        self.dense5 = keras.layers.Dense(64,
                                         kernel_regularizer=l2(1e-6),
                                         bias_regularizer=l1_l2(1e-5))
        self.row_out = keras.layers.Dense(dim, activation="linear")

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.pre(inputs, training=training)

        # Unpack given states if any, else initialize None
        if states is not None:
            s1, s2, s3 = states
        else:
            s1 = s2 = s3 = None

        # Pass through first LSTM layer
        x1, h1, c1 = self.rnn1(x, initial_state=s1, training=training)
        # BatchNorm applied to output of first LSTM
        xr = self.bn(x1, training=training)
        # Pass through second LSTM layer
        xr, h2, c2 = self.rnn2(xr, initial_state=s2, training=training)
        # Residual add output of first LSTM and processed second LSTM
        x_res = self.res([x1, xr])
        # Pass through third LSTM layer (no return_sequences)
        x_out, h3, c3 = self.rnn3(x_res, initial_state=s3, training=training)

        # First output branch with classification sigmoid output
        y = self.dense1(x_out, training=training)
        y = self.bn2(y, training=training)
        y = self.dense2(y, training=training)
        y = self.hosp_out(y, training=training)

        # Second output branch (presumably regression output)
        z = self.dense4(x_out, training=training)
        z = self.dropout3(z, training=training)
        z = self.dense5(z, training=training)
        z = self.row_out(z, training=training)

        if return_state:
            # Return outputs and states as tuple of lists
            return y, z, ([h1, c1], [h2, c2], [h3, c3])
        else:
            return y, z


def my_model_function():
    # Assume input feature dimension = 5 as from original example
    # Create and return an instance of MyModel with dim=5
    return MyModel(dim=5)


def GetInput():
    # Return a random float32 tensor of shape (Batch, TimeSteps, Features)
    # Batch size = 4 and time steps = 10 chosen arbitrarily
    B = 4
    T = 10
    F = 5  # features
    return tf.random.uniform((B, T, F), dtype=tf.float32)

