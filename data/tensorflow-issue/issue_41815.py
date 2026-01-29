# tf.random.uniform((B, T, 5), dtype=tf.int32) ← Input shape is [batch, time, 5] with int32 dtype based on the example

import tensorflow as tf

class TestRNNCell(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.units = 10
        self.state_size = 20

    def call(self, indices, state):
        # Assert input dtype is int32 as expected
        tf.debugging.assert_type(indices, tf.int32)
        # The operation on indices, e.g. gather looks like:
        output = tf.gather(tf.range(5), indices)
        # The state is unchanged (pass-through), assuming shape [batch, 20]
        return output, state

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Important: set dtype=tf.int32 on RNN layer to avoid SavedModel export dtype issue.
        self.rnn = tf.keras.layers.RNN(TestRNNCell(), dtype=tf.int32)

    @tf.function
    def call(self, indices):
        # Assert input dtype as int32
        tf.debugging.assert_type(indices, tf.int32)
        return self.rnn(indices)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random int32 tensor of shape [batch, time, 5]
    # Here batch=10, time=10, last dim=5 as per example
    return tf.random.uniform(
        shape=(10, 10, 5), minval=0, maxval=5, dtype=tf.int32
    )

