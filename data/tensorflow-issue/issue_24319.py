# tf.random.uniform((BATCH_SIZE, None, UNITS_INPUT_OUTPUT), dtype=tf.float32)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, count_lstms=200, batch_size=100, units_input_output=5, units_lstms=20):
        super().__init__()
        self.count_lstms = count_lstms
        self.batch_size = batch_size
        self.units_input_output = units_input_output
        self.units_lstms = units_lstms

        # Input layer specification to set batch shape
        # Note: None in sequence length dimension to allow variable length sequences
        self.input_layer = tf.keras.layers.InputLayer(
            batch_input_shape=(batch_size, None, units_input_output)
        )

        # Create multiple stateful LSTM layers sharing the same input tensor specification
        self.lstms = [
            tf.keras.layers.LSTM(
                units=units_lstms, 
                stateful=True, 
                return_sequences=True
            ) for _ in range(count_lstms)
        ]

        # A Dense layer applied via TimeDistributed to combined outputs
        self.time_distributed_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units_input_output)
        )

        # Concatenation axis is last feature dimension
        self.concat_axis = -1

    def call(self, inputs, training=False):
        # Pass input through input layer to enforce batch shape
        x = self.input_layer(inputs)

        # Run input through each LSTM layer independently
        lstm_outputs = []
        for lstm in self.lstms:
            lstm_out = lstm(x, training=training)
            lstm_outputs.append(lstm_out)

        # Concatenate LSTM outputs along the feature axis
        concat_output = tf.keras.layers.concatenate(lstm_outputs, axis=self.concat_axis)

        # Apply time distributed dense layer to get final output
        output = self.time_distributed_dense(concat_output)

        return output

    def reset_states(self):
        # Override reset_states to reset each LSTM layer's states
        for lstm in self.lstms:
            lstm.reset_states()


def my_model_function():
    # Create and return an instance of MyModel with default parameters
    return MyModel()


def GetInput():
    # Return a random tensor with shape matching expected input:
    # (batch_size, sequence_length, features) = (100, 1, 5) as per example
    batch_size = 100
    sequence_length = 1
    units_input_output = 5
    # Use uniform float32 tensor similar to example's np.random.randn, but scaled
    return tf.random.uniform(
        shape=(batch_size, sequence_length, units_input_output), dtype=tf.float32
    )

