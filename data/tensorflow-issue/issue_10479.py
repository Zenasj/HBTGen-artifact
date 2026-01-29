# tf.random.normal((5, 3, 7), dtype=tf.float32) ← Inferred input shape and dtype from original example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM cell with peepholes and orthogonal initializer replaced for peephole weights
        # Due to the original bug, we initialize peepholes separately with ones initializer
        # and rest of weights with orthogonal_initializer

        # We define the cell size matching the original example
        self.num_units = 7

        # Custom initializer for peephole: peephole weights are 1D, so orthogonal init fails.
        # Use ones initializer instead for peepholes.
        def custom_initializer(shape, dtype=None, partition_info=None):
            # If shape is 1D vector for peepholes, return ones; else orthogonal
            if len(shape) == 1:
                return tf.ones(shape, dtype=dtype)
            else:
                return tf.keras.initializers.Orthogonal()(shape, dtype=dtype)

        # Construct a Keras LSTMCell with peepholes using TensorFlow Addons since tf.contrib is deprecated.
        # However, since this is a reconstruction in TF 2.20, re-implement peephole manually.
        # For faithful reproduction, we simulate peephole logic inside a custom cell.

        # Because TF 2.x does not have built-in LSTMCell with peepholes,
        # we implement a simple peephole-enabled LSTMCell module manually for demonstration.

        self.cell = PeepholeLSTMCell(self.num_units,
                                     kernel_initializer=tf.keras.initializers.Orthogonal(),
                                     bias_initializer='zeros',
                                     peephole_initializer=tf.keras.initializers.Ones())

        # We define a tf.keras.layers.RNN wrapper with time_major=True to match original inputs
        self.rnn_layer = tf.keras.layers.RNN(self.cell,
                                             time_major=True,
                                             return_sequences=True,
                                             return_state=True)

    def call(self, inputs):
        # inputs shape: (time_steps=5, batch=3, input_dim=7)
        # The original example's inputs had shape [5,3,7]
        # The RNN returns outputs and the final states
        outputs_and_states = self.rnn_layer(inputs)
        # outputs_and_states: outputs plus cell and hidden states
        outputs = outputs_and_states[0]
        states = outputs_and_states[1:]
        return outputs, states


class PeepholeLSTMCell(tf.keras.layers.Layer):
    """
    LSTMCell with peephole connections.
    Adapted for TF 2.x, implements peephole connections where
    the cell state is connected to gates via learnable diagonal matrices.
    """

    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 peephole_initializer='ones',
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = [self.units, self.units]  # c and h
        self.output_size = self.units

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.peephole_initializer = tf.keras.initializers.get(peephole_initializer)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Weights for input and recurrent kernel
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            initializer=self.kernel_initializer,
            name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            initializer=self.kernel_initializer,
            name='recurrent_kernel')
        self.bias = self.add_weight(
            shape=(self.units * 4,),
            initializer=self.bias_initializer,
            name='bias')

        # Peephole weights - 3 vectors, diagonal multiplication with cell state
        self.w_f_diag = self.add_weight(
            shape=(self.units,), initializer=self.peephole_initializer, name='w_f_diag')
        self.w_i_diag = self.add_weight(
            shape=(self.units,), initializer=self.peephole_initializer, name='w_i_diag')
        self.w_o_diag = self.add_weight(
            shape=(self.units,), initializer=self.peephole_initializer, name='w_o_diag')

        super().build(input_shape)

    def call(self, inputs, states):
        prev_c, prev_h = states

        z = tf.matmul(inputs, self.kernel) + tf.matmul(prev_h, self.recurrent_kernel) + self.bias

        z0, z1, z2, z3 = tf.split(z, num_or_size_splits=4, axis=1)

        # Peephole connections added here
        f = tf.sigmoid(z0 + self.w_f_diag * prev_c)
        i = tf.sigmoid(z1 + self.w_i_diag * prev_c)
        o = tf.sigmoid(z3 + self.w_o_diag * prev_c)
        c_bar = tf.tanh(z2)

        c = f * prev_c + i * c_bar
        h = o * tf.tanh(c)

        return h, [c, h]


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input matching (time_steps=5, batch=3, input_dim=7)
    # Use float32 dtype to match original example
    return tf.random.normal(shape=(5, 3, 7), dtype=tf.float32)

