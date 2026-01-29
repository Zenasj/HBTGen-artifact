# tf.random.uniform((B, 7, 8), dtype=tf.float32) ← Input shape inferred from the FedMCRNNModel's X_SHAPE = [7,8]

import tensorflow as tf
from tensorflow import keras


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the model architecture in the issue:

        # First LSTM layer with 384 units,
        # input_shape=(7,8), return sequences True
        self.lstm1 = keras.layers.LSTM(
            units=384,
            return_sequences=True,
            input_shape=(7, 8),
            name="lstm1",
        )
        self.act1 = keras.layers.LeakyReLU(alpha=0.523629795960645)
        self.drop1 = keras.layers.Dropout(rate=0.372150795833)

        # Second LSTM layer with 64 units, return sequences True
        self.lstm2 = keras.layers.LSTM(
            units=64,
            return_sequences=True,
            name="lstm2",
        )
        self.act2 = keras.layers.LeakyReLU(alpha=0.523629795960645)
        self.drop2 = keras.layers.Dropout(rate=0.372150795833)

        # Third LSTM layer with 480 units, return sequences True
        self.lstm3 = keras.layers.LSTM(
            units=480,
            return_sequences=True,
            name="lstm3",
        )
        self.act3 = keras.layers.LeakyReLU(alpha=0.523629795960645)
        self.drop3 = keras.layers.Dropout(rate=0.372150795833)

        # Flatten and Dense output 1 unit (regression output)
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(1, name="output_dense")

        # Model compile arguments from original:
        self.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00668472266354),
            loss="mean_squared_error",
            metrics=["mean_absolute_error"],
        )

    def call(self, inputs, training=False):
        """
        Forward pass. 
        training flag controls dropout behavior.
        """
        x = self.lstm1(inputs)
        x = self.act1(x)
        x = self.drop1(x, training=training)

        x = self.lstm2(x)
        x = self.act2(x)
        x = self.drop2(x, training=training)

        x = self.lstm3(x)
        x = self.act3(x)
        x = self.drop3(x, training=training)

        x = self.flatten(x)
        x = self.dense(x)  # final shape [batch, 1]
        return x


def my_model_function():
    # Return an initialized instance of MyModel.
    # Weights will be randomly initialized here as in the original
    return MyModel()


def GetInput():
    # Returns a random tensor matching the input to MyModel
    # Shape (batch_size, 7, 8), dtype float32.
    # Batch size is arbitrary; choose 1 for simplicity.
    batch_size = 1
    return tf.random.uniform(shape=(batch_size, 7, 8), dtype=tf.float32)

