# tf.random.uniform((1, 3, 4), dtype=tf.float32) ← Input shape derived from BATCH_SIZE=1, NUM_SAMPLES=3, SAMPLE_SIZE=4

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM layer with 2 units returning sequences for each time step
        self.lstm = tf.keras.layers.LSTM(
            units=2,
            return_sequences=True,
        )
        # Flatten layer to produce final output shape (batch_size, time_steps * units)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=False):
        x = self.lstm(inputs, training=training)
        x = self.flatten(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor matching input expected by MyModel
    # Shape: (BATCH_SIZE=1, NUM_SAMPLES=3, SAMPLE_SIZE=4)
    # Uniform distribution in range [-1, 1] to match dataset_example generator
    return tf.random.uniform(shape=(1, 3, 4), minval=-1, maxval=1, dtype=tf.float32)

