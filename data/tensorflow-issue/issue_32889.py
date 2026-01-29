# tf.random.uniform((B=17, H=10, W=4), dtype=tf.float32) ← inferred input shape (batch_size=17, seq_len=10, seq_width=4)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, rnn_dim=64, rnn_layer_num=2, seq_len=10, seq_width=4, batch_size=17):
        super().__init__()
        self.rnn_dim = rnn_dim              # LSTM hidden dimension
        self.rnn_layer_num = rnn_layer_num  # Number of LSTM layers
        self.seq_len = seq_len              # Sequence length
        self.seq_width = seq_width          # Width of each sequence element
        self.batch_size = batch_size

        # Layers
        self._dense_in = tf.keras.layers.Dense(self.rnn_dim)
        self._cells = {}
        for i in range(self.rnn_layer_num):
            self._cells[i] = tf.keras.layers.LSTMCell(units=self.rnn_dim)
        self._dens_out = tf.keras.layers.Dense(self.seq_width, activation="sigmoid")

        # Initialize cell states in build
        self.states = None

    def build(self, input_shape):
        # Initialize LSTM cell states as variables to hold batch state between calls
        # Each state is a tuple of (h, c) with shape [batch_size, rnn_dim]
        init_states = {}
        for i in range(self.rnn_layer_num):
            h = tf.Variable(tf.random.uniform([self.batch_size, self.rnn_dim]), trainable=False)
            c = tf.Variable(tf.random.uniform([self.batch_size, self.rnn_dim]), trainable=False)
            init_states[i] = [h, c]
        self.states = init_states
        super().build(input_shape)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=None):
        # inputs shape: (batch_size, 1, rand_input_dim)
        inputs = tf.squeeze(inputs, axis=1)  # Now shape (batch_size, rand_input_dim)

        # Use the stored states; copy their values for this call
        states = {}
        for i in range(self.rnn_layer_num):
            h_var, c_var = self.states[i]
            states[i] = (h_var.read_value(), c_var.read_value())

        rand_prev = tf.random.uniform([self.batch_size, self.seq_width], dtype=tf.float32)
        prev_inputs = tf.concat([rand_prev, inputs], axis=-1)
        outputs = []

        for _ in tf.range(self.seq_len):
            cell_inputs = self._dense_in(prev_inputs)  # Shape (batch_size, rnn_dim)
            new_states = {}

            for i, cell in self._cells.items():
                output, new_state = cell(cell_inputs, states[i], training=training)
                cell_inputs = output  # input for next layer
                new_states[i] = new_state

            step_outputs = self._dens_out(output)  # Shape (batch_size, seq_width)
            prev_inputs = tf.concat([step_outputs, inputs], axis=-1)
            outputs.append(tf.expand_dims(step_outputs, axis=1))
            states = new_states

        # Update stored states variables with new states values
        for i in range(self.rnn_layer_num):
            h_var, c_var = self.states[i]
            h_val, c_val = states[i]
            h_var.assign(h_val)
            c_var.assign(c_val)

        return tf.concat(outputs, axis=1)  # Shape (batch_size, seq_len, seq_width)

def my_model_function():
    # Return an instance of MyModel with fixed parameters from example
    return MyModel(rnn_dim=64, rnn_layer_num=2, seq_len=10, seq_width=4, batch_size=17)

def GetInput():
    # Return a random input tensor matching model's expected input shape: (batch_size, 1, input_dim)
    # From the original example, input dimension is 3 for rand_input_dim
    batch_size = 17
    input_dim = 3
    return tf.random.uniform((batch_size, 1, input_dim), dtype=tf.float32)

