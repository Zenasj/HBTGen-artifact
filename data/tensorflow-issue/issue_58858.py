# tf.random.normal((1, 1, 192), dtype=tf.float32) ← Input shape inferred from original code: batch size 1, time step 1, feature dim 192

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # BatchNormalization on input with feature dim = 192
        self.bn = tf.keras.layers.BatchNormalization()
        # LSTMCell with units equal to feature dimension 192
        self.lstm_cell = tf.keras.layers.LSTMCell(units=192)

    def call(self, inputs, states):
        """
        inputs: Tensor of shape [batch_size, 1, 192] (time step dimension = 1)
        states: list of two tensors [state_h, state_c], each of shape [batch_size, 192]
        
        Returns:
        - out: x + xT, where x is output from LSTMCell, xT is batch normalized input
        - new_state_h: hidden state output of LSTMCell
        - new_state_c: cell state output of LSTMCell
        """
        inp = inputs  # [B, 1, 192]
        state_h, state_c = states    # each [B, 192]

        # Apply batch normalization along feature axis - inputs is 3D [B, 1, 192]
        xT = self.bn(inp)  # [B, 1, 192]

        # Activation tanh on batch normalized input
        lstm_in = tf.keras.activations.tanh(xT)

        # LSTMCell expects input with shape [batch_size, feature_dim], so we remove time dim by lstm_in[:,0,:]
        x, new_states = self.lstm_cell(lstm_in[:, 0, :], states=[state_h, state_c])  # x: [B, 192]

        new_state_h, new_state_c = new_states

        # xT is [B, 1, 192], so squeeze time dim for addition with x
        xT_squeezed = tf.squeeze(xT, axis=1)  # [B, 192]

        # Output is element-wise addition of LSTMCell output and batch normalized input
        out = x + xT_squeezed  # [B, 192]

        return out, new_state_h, new_state_c

def my_model_function():
    # Create and return an instance of MyModel with batchnorm and LSTMCell as described
    return MyModel()

def GetInput():
    # Return the input tuple required by MyModel:
    # inputs tensor with shape [1, 1, 192] (batch size 1, time 1, features 192)
    # initial state_h and state_c tensors each [1, 192]
    input_tensor = tf.random.normal([1, 1, 192], dtype=tf.float32)
    init_state_h = tf.zeros([1, 192], dtype=tf.float32)
    init_state_c = tf.zeros([1, 192], dtype=tf.float32)
    return input_tensor, [init_state_h, init_state_c]

