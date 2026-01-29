# tf.random.uniform((3, 3, 5), dtype=tf.float32), constants shape (3, 3)

import tensorflow as tf

class RNNCellWithConstants(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # The state size of the cell is 5
        self.state_size = 5
        super(RNNCellWithConstants, self).__init__(**kwargs)

    def build(self, input_shape):
        # The input_shape is expected to be a list of shapes:
        # [input_timestep_shape, constants_shape]
        # For example: [(3, 5), (3, 3)]
        # We print shape for debugging, and mark the layer built
        # Usually the RNN layer passes the shape for a single timestep of input (batch, input_size)
        print("Build input shapes:", input_shape)
        self.built = True

    def call(self, inputs, states, constants):
        # inputs: shape (batch, input_size) for a single timestep
        # states: list of previous state(s), each of shape self.state_size
        # constants: additional tensor passed through RNN
        # This example just returns inputs as output and sets new states = [inputs]
        # Prints are for debugging purposes as in original code.
        print("Call inputs:", inputs)
        print("Call states:", states)
        print("Call constants:", constants)
        return inputs, [inputs]

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cell = RNNCellWithConstants()
        # The tf.keras.layers.RNN wraps the cell; it handles sequencing through time.
        self.rnn_layer = tf.keras.layers.RNN(self.cell)

    def call(self, inputs_and_constants):
        # inputs_and_constants expected as tuple (inputs, constants)
        inputs, constants = inputs_and_constants
        # Call the RNN layer with constants argument
        return self.rnn_layer(inputs, constants=constants)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tuple of two tensors (inputs, constants) matching the expected inputs:
    # inputs shape: (batch=3, timesteps=3, features=5), dtype float32
    # constants shape: (batch=3, const_features=3), dtype float32
    inputs = tf.random.uniform((3, 3, 5), dtype=tf.float32)
    constants = tf.random.uniform((3, 3), dtype=tf.float32)
    return (inputs, constants)

