# tf.random.uniform((16, 43, 256), dtype=tf.float32) ← inferred input shape from the issue "model(tf.random.uniform([16,43,256]))"

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Inputs: sequences of shape (batch, steps, features)
        # From issue: input_length=256, batch=16, steps=43
        self.hidden_size = 64
        
        # LSTM layer with return_sequences=True and return_state=True to provide output sequence and hidden/cell states
        self.lstm = tf.keras.layers.LSTM(
            self.hidden_size,
            return_sequences=True,
            return_state=True,
            input_shape=(None, 256)  # time-major False (batch, time, features)
        )
        
        # Dense layer maps hidden dimension to 1 output feature per timestep
        self.dense = tf.keras.layers.Dense(1)

    def call(self, x):
        # Forward pass through LSTM
        output_seq, h_state, c_state = self.lstm(x)  
        # output_seq shape: (batch, time_steps, hidden_size)
        
        # Apply dense layer to each timestep output (broadcasted)
        output = self.dense(output_seq)  # shape (batch, time_steps, 1)
        
        # Return both output sequence and stacked hidden states [h_state, c_state]
        # This keeps compatibility with the issue requirement to "return x, tf.stack([h0, c0])"
        states = tf.stack([h_state, c_state])
        return output, states

def my_model_function():
    # Create a new instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 tensor matching input shape expected by MyModel
    # Batch size, time steps, features = (16, 43, 256)
    return tf.random.uniform((16, 43, 256), dtype=tf.float32)

