# tf.random.uniform((32, 2, 20), dtype=tf.float32) ← Based on placeholder shape used for inputs

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, state_size=20):
        super(MyModel, self).__init__()
        self.state_size = state_size
        # Using tf.keras.layers.Dense instead of deprecated tf.layers.dense
        self.dense_layer = tf.keras.layers.Dense(self.state_size, activation=None, name="encoder_dense")
        
        # In original TF1 code, a tf.make_template called "encoder" wraps a dense op:
        # We'll emulate this with a simple dense layer called via a method.
        # There is no update ops for this layer, consistent with the fix in issue.
    
    def encoder(self, prev_state, obs):
        # Concatenate prev_state and current observation on last dimension
        x = tf.concat([prev_state, obs], axis=-1)
        return self.dense_layer(x)
    
    def call(self, inputs):
        """
        inputs: Tensor of shape [batch_size, time_steps, features]
        Run a custom RNN loop similar to tf.nn.dynamic_rnn with custom cell.
        
        The cell state and output size = self.state_size
        """
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        # Initialize state to zeros: shape [batch_size, state_size]
        state = tf.zeros([batch_size, self.state_size], dtype=inputs.dtype)
        
        outputs = []
        # Loop over time dimension
        for t in range(time_steps):
            input_t = inputs[:, t, :]
            # cell call: next state and output (output = state)
            state = self.encoder(state, input_t)
            outputs.append(state)
        
        # Stack outputs along time axis: shape [batch_size, time_steps, state_size]
        outputs = tf.stack(outputs, axis=1)
        return outputs, state

def my_model_function():
    # Return an instance of MyModel with state_size=20 as per original code
    return MyModel(state_size=20)

def GetInput():
    # Return a random float32 tensor shaped to [32, 2, 20]
    # Corresponds to batch of 32, sequence length 2, features 20
    return tf.random.uniform((32, 2, 20), dtype=tf.float32)

