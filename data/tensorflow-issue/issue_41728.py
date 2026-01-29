# tf.random.uniform((B, T, D), dtype=tf.float32)  ← Assumed input shape: batch_size B, time_steps T, feature_dim D = input dimension compatible with GRUCell

import tensorflow as tf
from tensorflow import keras

class GRUCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        # state_size and output_size must be int or TensorShape for RNNCells
        self.state_size = units   # as integer
        self.output_size = units  # as integer
        super(GRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = (batch_size, input_dim)
        self.dim = input_shape[-1]

        self.w_r = self.add_weight(shape=[self.dim + self.units, self.units], 
                                   initializer='uniform', name='reset_gate', trainable=True)
        self.b_r = self.add_weight(shape=[self.units], initializer='zeros', 
                                   name='reset_gate_bias', trainable=True)

        self.w_z = self.add_weight(shape=[self.dim + self.units, self.units], 
                                   initializer='uniform', name='update_gate', trainable=True)
        self.b_z = self.add_weight(shape=[self.units], initializer='zeros', 
                                   name='update_gate_bias', trainable=True)

        self.w_n = self.add_weight(shape=[self.dim + self.units, self.units], 
                                   initializer='uniform', name='interim_gate', trainable=True)
        self.b_n = self.add_weight(shape=[self.units], initializer='zeros', 
                                   name='interim_gate_bias', trainable=True)

        super(GRUCell, self).build(input_shape)

    def call(self, inputs, states):
        # inputs shape: (batch_size, input_dim)
        # states is list of previous states (length 1), shape (batch_size, units)
        prev_state = states[0]

        # Concatenate inputs and state parts for each gate calculation as slices:
        # Equivalent to: inputs @ W_x + states @ W_h + bias
        # But weights combined as [input_dim + units, units]
        x_r = inputs @ self.w_r[:self.dim, :]  # input part for reset gate
        h_r = prev_state @ self.w_r[self.dim:, :]  # hidden part for reset gate
        r = tf.nn.sigmoid(x_r + h_r + self.b_r)

        x_z = inputs @ self.w_z[:self.dim, :]
        h_z = prev_state @ self.w_z[self.dim:, :]
        z = tf.nn.sigmoid(x_z + h_z + self.b_z)

        x_n = inputs @ self.w_n[:self.dim, :]
        h_n = (prev_state * r) @ self.w_n[self.dim:, :]
        n = tf.nn.tanh(x_n + h_n + self.b_n)

        output = (1 - z) * prev_state + z * n

        return output, [output]

    def get_config(self):
        config = super(GRUCell, self).get_config()
        config.update({"units": self.units})
        return config

class MyModel(tf.keras.Model):
    def __init__(self, units=32):
        super(MyModel, self).__init__()
        # Using the custom GRUCell wrapped inside keras.layers.RNN with return_sequences=False
        self.rnn = keras.layers.RNN(GRUCell(units))
        self.dense = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.rnn(inputs)
        return self.dense(x)

def my_model_function():
    # Create MyModel instance with default units=32
    model = MyModel(units=32)
    # Explicitly build the model by calling it on a sample input with fixed batch and time dims.
    # This ensures model.built is True and is compatible with quantization workflows.
    sample_input = tf.random.uniform((1, 10, 8), dtype=tf.float32)
    model(sample_input)  # build model; call once to create weights
    # Compile to mimic behavior shown in issue (also helps some workflows)
    model.compile(optimizer=keras.optimizers.RMSprop(),
                  loss='mae', metrics=['mse'])
    return model

def GetInput():
    # Provide a random input matching shape (batch_size=4, time_steps=10, features=8)
    # These dimensions are assumed reasonable for the custom GRUCell input
    return tf.random.uniform((4, 10, 8), dtype=tf.float32)

