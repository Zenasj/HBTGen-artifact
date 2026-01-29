# tf.random.normal((6, 5, 10), dtype=tf.float32) ← Input shape: batch=6, timesteps=5, features=10

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Example sizes inferred from the issue description
        rnn_units = 10
        dense_units = 8

        self.r1 = tf.keras.layers.SimpleRNN(rnn_units)
        self.flat = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(rnn_units)
        self.r2 = tf.keras.layers.SimpleRNN(rnn_units)
        self.d2 = tf.keras.layers.Dense(dense_units)

    def call(self, inputs, **kwargs):
        """
        The call method demonstrates using SimpleRNN with and without initial state.
        inputs: tensor of shape (batch, timesteps, features)
        """

        # First SimpleRNN layer output (no initial state)
        x = self.r1(inputs)  # shape: (batch, rnn_units)

        # Compute a state vector from flattened x, to use as initial state for second RNN
        state = self.d1(self.flat(x))  # shape: (batch, rnn_units)

        # Second SimpleRNN uses inputs plus initial state
        # As per documented usage, input can be [inputs, initial_state]
        # initial_state should be a list of tensors matching the expected state of rnn_units
        x = self.r2([inputs, state])

        # Final dense layer
        x = self.d2(x)  # shape: (batch, dense_units)

        return x

def my_model_function():
    # Return an instance of MyModel as defined above
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the input expected by MyModel
    # From the example: batch=6, timesteps=5, features=10
    return tf.random.normal(shape=(6, 5, 10), dtype=tf.float32)

