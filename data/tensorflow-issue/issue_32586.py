# tf.random.uniform((4, 7, 16), dtype=tf.float32) ← inferred input shape from example

import tensorflow as tf

class CellWrapper(tf.keras.layers.AbstractRNNCell):
    def __init__(self, cell):
        super(CellWrapper, self).__init__()
        self.cell = cell

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self.cell.output_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.cell.get_initial_state(
            inputs=inputs, batch_size=batch_size, dtype=dtype)

    # The fix: forward the training argument down to wrapped cell
    def call(self, inputs, states, training=None, **kwargs):
        # According to the issue, training should not be None here when called during training.
        assert training is not None, "training flag must be passed to CellWrapper.call()"
        return self.cell.call(inputs, states, training=training, **kwargs)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Create an LSTMCell wrapped in CellWrapper, then stacked with StackedRNNCells
        cell = tf.keras.layers.LSTMCell(32)
        wrapped_cell = CellWrapper(cell)
        stacked_cell = tf.keras.layers.StackedRNNCells([wrapped_cell])
        self.rnn = tf.keras.layers.RNN(stacked_cell)

    def call(self, inputs, training=None, **kwargs):
        # Pass training flag to RNN so it can be forwarded properly to cells
        return self.rnn(inputs, training=training)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the expected input (batch=4, timesteps=7, features=16)
    return tf.random.uniform(shape=(4, 7, 16), dtype=tf.float32)

