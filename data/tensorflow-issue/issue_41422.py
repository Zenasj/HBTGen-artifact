# tf.random.uniform((1, 7, 12), dtype=tf.float32)  ← Input shape inferred from TimeseriesGenerator with n_input=7 and 12 features

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        # We have two model architectures discussed in the issue:
        # 1) LSTM + Dense treating output as flattened vector (y reshaped to (None, 36))
        # 2) LSTM with return_sequences + MaxPool1D + Dense output
        
        # For fusion and comparison, we encapsulate both.
        # The input shape is (7, 12) and target shape (3, 12).
        
        self.n_input = 7
        self.n_features = 12
        self.output_days = 3
        
        # Model A: LSTM + Dense (flattened output)
        self.lstm_a = tf.keras.layers.LSTM(64, input_shape=(self.n_input, self.n_features))
        # Output layer to predict all features for 3 days flattened:
        self.dense_a = tf.keras.layers.Dense(self.output_days * self.n_features)
        
        # Model B: LSTM with return_sequences + MaxPool1D + Dense
        self.lstm_b = tf.keras.layers.LSTM(4, return_sequences=True, input_shape=(self.n_input, self.n_features))
        self.maxpool_b = tf.keras.layers.MaxPool1D(pool_size=2)
        # Dense output layer to produce feature dimension output, applied on each time step
        self.dense_b = tf.keras.layers.Dense(self.n_features)
        
    def call(self, inputs):
        # inputs shape: (batch, 7, 12)
        
        # Model A forward pass
        x_a = self.lstm_a(inputs)  # shape (batch, 64)
        out_a = self.dense_a(x_a)  # shape (batch, 36)
        # Reshape output back to (batch, 3, 12)
        out_a_reshaped = tf.reshape(out_a, (-1, self.output_days, self.n_features))
        
        # Model B forward pass
        x_b = self.lstm_b(inputs)  # shape (batch, 7, 4)
        x_b_pooled = self.maxpool_b(x_b)  # shape (batch, ~3, 4) maxpool 2 shrinks time dimension roughly by 2
        # To match target time dimension (3), we assume maxpool reduces time dimension close to 3.
        # Apply Dense to each time-step to output (batch, time_steps, features)
        out_b = self.dense_b(x_b_pooled)  # shape (batch, 3, 12) (approx.)
        
        # To fuse outputs for comparison, compute element-wise absolute difference:
        # Since the dimension might technically differ slightly due to pooling, 
        # but maxpool with pool_size=2 on 7 steps results (floor((7-2)/2)+1)=3 steps,
        # so shapes are compatible (batch,3,12).
        
        diff = tf.abs(out_a_reshaped - out_b)  # shape (batch, 3, 12)
        
        # Return the difference tensor, could be used as a metric for model agreement or debugging.
        # Optionally, could also return boolean 'close' tensor or concatenated outputs.
        return diff

def my_model_function():
    return MyModel()

def GetInput():
    # According to the problem, typical input shape is (batch_size, 7, 12)
    # Generate a random float32 tensor of this shape
    return tf.random.uniform((1, 7, 12), dtype=tf.float32)

