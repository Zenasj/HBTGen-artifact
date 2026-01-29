# tf.random.uniform((1, 10, 1), dtype=tf.float32) ← inferred input shape from example batch_shape=(batch_size=1, sample_size=10, n_channels=1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, stateful=False, batch_size=None):
        super().__init__()
        self.n_channels = 1
        self.sample_size = 10
        self.n_units = 5
        self.stateful = stateful
        self.batch_size = batch_size

        # Define LSTM layer with stateful or stateless config
        self.lstm = tf.keras.layers.LSTM(
            units=self.n_units,
            activation='tanh',
            recurrent_activation='sigmoid',
            return_sequences=False,
            stateful=self.stateful,
            batch_input_shape=(self.batch_size, self.sample_size, self.n_channels) 
                if self.stateful else (None, self.sample_size, self.n_channels)
        )

        # Dense layer to output single value
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        output = self.dense(x)
        return output


def my_model_function():
    # Return an instance of the model with stateful=True and batch_size=1,
    # as in the original reproducible example.
    # Stateful LSTM requires fixed batch size.
    return MyModel(stateful=True, batch_size=1)


def GetInput():
    # Return a random input tensor matching expected input shape:
    # (batch_size=1, sample_size=10, n_channels=1)
    return tf.random.uniform(shape=(1, 10, 1), dtype=tf.float32)

