# tf.random.uniform((batch_size, timestep, i1), dtype=tf.float32), tf.random.uniform((batch_size, timestep, i2, i3), dtype=tf.float32), tf.random.uniform((batch_size, timestep, i1), dtype=tf.float32)

import tensorflow as tf
from tensorflow import keras

class NestedCell(keras.layers.Layer):
    def __init__(self, unit_1, unit_2, unit_3, **kwargs):
        super(NestedCell, self).__init__(**kwargs)
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.unit_3 = unit_3
        # State and output sizes are nested Tensors shapes
        self.state_size = [tf.TensorShape([unit_1]), tf.TensorShape([unit_2, unit_3])]
        self.output_size = [tf.TensorShape([unit_1]), tf.TensorShape([unit_2, unit_3])]

    def build(self, input_shapes):
        # Inputs expected: list/tuple of two tensors shapes
        # input_shapes[0]: (batch_size, i1)
        # input_shapes[1]: (batch_size, i2, i3)
        i1 = input_shapes[0][1]
        i2 = input_shapes[1][1]
        i3 = input_shapes[1][2]

        # Weight kernel_1: (i1, unit_1)
        self.kernel_1 = self.add_weight(
            shape=(i1, self.unit_1),
            initializer="uniform",
            name="kernel_1"
        )
        # Weight kernel_2_3: (i2, i3, unit_2, unit_3)
        self.kernel_2_3 = self.add_weight(
            shape=(i2, i3, self.unit_2, self.unit_3),
            initializer="uniform",
            name="kernel_2_3"
        )
        super(NestedCell, self).build(input_shapes)

    def call(self, inputs, states, constants=None):
        # inputs is a tuple/list of two tensors:
        #   input_1: (batch, i1)
        #   input_2: (batch, i2, i3)
        # states is a list/tuple of two tensors:
        #   state_1: (batch, unit_1)
        #   state_2_3: (batch, unit_2, unit_3)
        # constants is not used in this example but included for signature compatibility
        input_1, input_2 = tf.nest.flatten(inputs)
        s1, s2 = states

        output_1 = tf.matmul(input_1, self.kernel_1)  # shape (batch, unit_1)
        # einsum to contract input_2 with kernel_2_3
        output_2_3 = tf.einsum("bij,ijkl->bkl", input_2, self.kernel_2_3)  # (batch, unit_2, unit_3)

        state_1 = s1 + output_1
        state_2_3 = s2 + output_2_3

        output = (output_1, output_2_3)
        new_states = (state_1, state_2_3)

        return output, new_states

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # Provide initial states filled with zeros
        state_1 = tf.zeros([batch_size, self.unit_1], dtype=dtype)
        state_2_3 = tf.zeros([batch_size, self.unit_2, self.unit_3], dtype=dtype)
        return [state_1, state_2_3]

    def get_config(self):
        return {"unit_1": self.unit_1, "unit_2": self.unit_2, "unit_3": self.unit_3}


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Units from original example
        self.unit_1 = 10
        self.unit_2 = 20
        self.unit_3 = 30

        # Create the custom NestedCell
        self.cell = NestedCell(self.unit_1, self.unit_2, self.unit_3)

        # Create the RNN layer wrapping the custom cell.
        # We specify that the cell expects constants argument by passing it to RNN layer
        self.rnn = keras.layers.RNN(
            self.cell,
            return_sequences=True,
            # Note: Keras RNN layer accepts 'constants' only in functional API or call
        )

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs: a tuple of (input_1, input_2, input_const)
        # But the cell does not actually use 'constants' in computation here.
        input_1, input_2, input_const = inputs

        # The keras RNN layer supports passing constants argument to the call method,
        # so we forward constants as a kwarg.
        outputs = self.rnn(
            inputs=(input_1, input_2),
            constants=input_const
        )
        return outputs


def my_model_function():
    # Instantiate and return MyModel
    return MyModel()


def GetInput():
    # Construct a sample random input matching the model expectation:
    # Shapes:
    # input_1: (batch_size, timestep, i1) = (64, 50, 32)
    # input_2: (batch_size, timestep, i2, i3) = (64, 50, 64, 32)
    # input_const: (batch_size, timestep, i1) = (64, 50, 32)
    batch_size = 64
    timestep = 50
    i1 = 32
    i2 = 64
    i3 = 32

    input_1 = tf.random.uniform((batch_size, timestep, i1), dtype=tf.float32)
    input_2 = tf.random.uniform((batch_size, timestep, i2, i3), dtype=tf.float32)
    input_const = tf.random.uniform((batch_size, timestep, i1), dtype=tf.float32)

    return (input_1, input_2, input_const)

