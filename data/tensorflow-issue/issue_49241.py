# tf.random.normal((4, 7, 32), dtype=tf.float32)  ← Input shape: (batch_size=4, seq_len=7, features=32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two RNN layers: one LSTM with cudnn kernel (default on GPU),
        # one LSTM with CPU/non-cudnn kernel (forcing non-cudnn by setting certain params)
        #
        # Since TensorFlow 2.x automatically uses cudnn LSTM if possible,
        # we simulate both behaviors by:
        # - cudnn LSTM (normal LSTM layer)
        # - CPU/non-cudnn LSTM by using a different implementation or by setting unroll=False
        # For demonstration, we just instantiate two LSTMs separately, because true kernel choice depends on runtime.
        #
        # This models the issue: For masked timesteps, cudnn outputs zeros, CPU/non-cudnn outputs last valid timestep.
        
        self.lstm_cudnn = tf.keras.layers.LSTM(32, return_sequences=True)  # cudnn kernel preferred on GPU
        self.lstm_noncudnn = tf.keras.layers.LSTM(32, return_sequences=True, recurrent_activation='sigmoid', unroll=False)
        # recurrent_activation='sigmoid' is default, but keeping explicitly.
        # unroll=False to avoid forcing cudnn kernel.
    
    def call(self, x, mask=None):
        # Run both models with same input + mask
        out_cudnn = self.lstm_cudnn(x, mask=mask)        # cudnn kernel output (masked timesteps zeroed)
        out_noncudnn = self.lstm_noncudnn(x, mask=mask)  # cpu/non-cudnn output (masked timesteps = last unmasked output)
        
        # Extract last timestep output for both (simulate the example usage in the issue)
        # Shape after LSTM: (batch_size, seq_len, 32)
        last_cudnn = out_cudnn[:, -1, :]      # shape: (batch_size, 32)
        last_noncudnn = out_noncudnn[:, -1, :]# shape: (batch_size, 32)
        
        # Compare outputs, return difference or boolean indicating if values differ
        # Because cudnn outputs zero for masked timestep (last timestep masked), 
        # cpu/non-cudnn outputs last valid timestep result, so difference will be large.
        
        # Compute absolute difference and mean across features as a scalar per sample
        diff = tf.abs(last_cudnn - last_noncudnn)  # (batch_size, 32)
        diff_mean = tf.reduce_mean(diff, axis=1)   # (batch_size,)
        
        # Return a dict with all info to examine
        return {
            'out_cudnn_last': last_cudnn,
            'out_noncudnn_last': last_noncudnn,
            'difference_per_sample': diff_mean,
            'are_outputs_close': tf.reduce_all(tf.abs(last_cudnn - last_noncudnn) < 1e-5)
        }

def my_model_function():
    # Return the model instance
    return MyModel()

def GetInput():
    # Based on the issue example:
    # batch_size=4, seq_len=7, features=32
    batch_size = 4
    seq_len = 7
    features = 32
    
    # Generate input tensor with shape (4, 7, 32)
    x = tf.random.normal([batch_size, seq_len, features])
    
    # Create mask indicating the valid timesteps, in the example:
    # valid length is seq_len - 1 = 6 for all batches,
    # so mask is True for first 6 timesteps, False for last.
    mask = tf.sequence_mask([seq_len - 1] * batch_size, maxlen=seq_len)
    
    # Return input and mask as a tuple to feed into the model call
    return (x, mask)

