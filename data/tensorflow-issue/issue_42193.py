# tf.random.uniform((B, 10, 1), dtype=tf.float32) ← input shape inferred from example (batch_size=1, seq_len=10, channels=1)

import tensorflow as tf

class TestCell(tf.keras.layers.Layer):
    state_size = 1
    output_size = 1

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # Expected initial state is all ones, shape (batch_size, 1)
        return tf.ones((batch_size, 1), dtype=dtype)

    def call(self, inputs, states):
        # Assert that the state passed in is all ones, to catch if the initial state is zero (the reported issue)
        tf.debugging.assert_equal(states, tf.ones_like(states))
        # RNN cell outputs inputs as output, and returns states unmodified
        return inputs, states

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use stateful=True to reflect the reported problematic use case
        self.rnn = tf.keras.layers.RNN(TestCell(), stateful=True)

    def call(self, inputs):
        # The issue is that despite TestCell overriding get_initial_state, 
        # the initial state used is zero when stateful=True.
        # This model just runs the RNN layer on inputs.
        return self.rnn(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return random input matching expected input shape: (batch_size=1, seq_length=10, channels=1)
    # Use dtype float32 for compatibility with the model
    batch_size = 1
    seq_length = 10
    channels = 1
    return tf.random.uniform((batch_size, seq_length, channels), dtype=tf.float32)

