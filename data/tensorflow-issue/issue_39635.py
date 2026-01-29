# tf.random.uniform((seq_len, batch_size, feature_dim), dtype=tf.float32)
# Note: seq_len=2, batch_size=1, feature_dim=1 as per provided reproduction code

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Fix the bidirectional LSTM to properly handle time_major=True input by overriding
        # the direction reversal to apply on the sequence axis (axis=0),
        # rather than the batch axis (axis=1) as was the bug
        
        # We replicate a Bidirectional wrapper, but fix the reversing axis for backward layer manually.
        
        # Use LSTM layer with return_sequences=True, time_major=True
        self.forward_lstm = tf.keras.layers.LSTM(
            1, return_sequences=True, time_major=True, name="forward_lstm"
        )
        self.backward_lstm = tf.keras.layers.LSTM(
            1, return_sequences=True, time_major=True, go_backwards=True, name="backward_lstm"
        )
    
    def call(self, inputs, training=False):
        # inputs shape is expected to be [seq_len, batch_size, feature_dim], time_major=True
        
        # Forward output
        y_forward = self.forward_lstm(inputs, training=training)  # shape: [seq_len, batch_size, units]
        
        # Backward outputs computed by backward_lstm going backwards on sequence axis
        y_backward_rev = self.backward_lstm(inputs, training=training)  # reversed over seq axis internally
        
        # The original bug was reversing on incorrect axis before concatenation.
        # Fix: reverse the backward output sequence on axis=0 (time axis) to align with forward output
        y_backward = tf.reverse(y_backward_rev, axis=[0])
        
        # Concatenate forward and backward on the last dimension (units)
        output = tf.concat([y_forward, y_backward], axis=-1)
        return output

def my_model_function():
    # Return an instance of MyModel with default initialization.
    # Weights can be manually set later to all ones to reproduce the original test.
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # MyModel expects shape [seq_len, batch_size, feature_dim], time_major=True
    seq_len = 2
    batch_size = 1
    feature_dim = 1
    return tf.random.uniform((seq_len, batch_size, feature_dim), dtype=tf.float32)

