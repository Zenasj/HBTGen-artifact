# tf.random.uniform((B, 5, 5), dtype=tf.float32), tf.random.uniform((B, 5, 5), dtype=tf.float32), tf.random.uniform((B, 3), dtype=tf.float32)

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.cell = RNNCellWithConstants(units=32, constant_size=3)
        self.rnn = keras.layers.RNN(self.cell)

    def call(self, inputs):
        # inputs is a list/tuple of three tensors: x1, x2, constant
        x1, x2, constant = inputs
        # Pass (x1, x2) as a tuple as RNN input (multiple inputs).
        # Provide constants separately as required by the cell
        y = self.rnn((x1, x2), constants=constant)
        return y


class RNNCellWithConstants(keras.layers.Layer):
    def __init__(self, units, constant_size, **kwargs):
        self.units = units
        self.state_size = units
        self.constant_size = constant_size
        super(RNNCellWithConstants, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is a tuple/list of two shapes for x1 and x2
        # We only use x1 in kernel shapes according to original code
        self.input_kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            initializer='uniform',
            name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.constant_kernel = self.add_weight(
            shape=(self.constant_size, self.units),
            initializer='uniform',
            name='constant_kernel')
        self.built = True

    def call(self, inputs, states, constants):
        # inputs: tuple/list of [x1, x2] (x2 is unused in computation per original code)
        # states: previous output state list
        # constants: list with one tensor constant
        x1, _ = inputs
        prev_output = states[0]
        constant = constants[0]
        h_input = tf.keras.backend.dot(x1, self.input_kernel)
        h_state = tf.keras.backend.dot(prev_output, self.recurrent_kernel)
        h_const = tf.keras.backend.dot(constant, self.constant_kernel)
        output = h_input + h_state + h_const
        return output, [output]

    def get_config(self):
        config = {'units': self.units, 'constant_size': self.constant_size}
        base_config = super(RNNCellWithConstants, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def my_model_function():
    return MyModel()

def GetInput():
    # Inputs:
    # x1: shape (batch, time, 5)
    # x2: shape (batch, time, 5) (dummy, not used in cell computations)
    # constant: shape (batch, 3)
    batch = 6
    time = 5
    dim = 5
    const_dim = 3
    # Generate random uniform tensors with float32 dtype, as per example
    import tensorflow as tf
    x1 = tf.random.uniform((batch, time, dim), dtype=tf.float32)
    x2 = tf.random.uniform((batch, time, dim), dtype=tf.float32)
    c = tf.random.uniform((batch, const_dim), dtype=tf.float32)
    return [x1, x2, c]

