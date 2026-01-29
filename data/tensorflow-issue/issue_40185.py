# tf.random.uniform((B, T, 256), dtype=tf.float32) ← Input shape (Batch, Time, Features) with dynamic batch and time dims

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, GRU

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # This model encapsulates both an LSTM-based and a GRU-based submodel
        # to reflect the discussion of issues with dynamic shapes on LSTM but not GRU
        self.bilstm_model = Bidirectional(LSTM(4, time_major=False, return_sequences=True), name='BILSTM_OUTPUT')
        self.bistugru_model = Bidirectional(GRU(4, time_major=False, return_sequences=True), name='BIGRU_OUTPUT')

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 256], dtype=tf.float32)])
    def call(self, inputs):
        # Forward pass both models on the same input
        lstm_output = self.bilstm_model(inputs)
        gru_output = self.bistugru_model(inputs)

        # Compare outputs element-wise to identify discrepancies
        # The difference tensor:
        diff = tf.abs(lstm_output - gru_output)

        # For demonstration, consider outputs "close" if difference < 1e-5 (tolerance)
        tolerance = 1e-5
        close = tf.less_equal(diff, tolerance)

        # Return a dictionary of outputs and their comparison mask
        # This fusion reflects discussion about LSTM vs GRU behavior on dynamic inputs
        return {
            'lstm_output': lstm_output,
            'gru_output': gru_output,
            'outputs_close': close,
            'max_diff': tf.reduce_max(diff)
        }

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # IMPORTANT: Run a dummy forward pass to initialize weights before saving/conversion as suggested in the issue comments
    dummy_input = tf.random.uniform(shape=[1, 2, 256], dtype=tf.float32)
    model(dummy_input)
    return model

def GetInput():
    # Generates random input tensor with dynamic batch and time dimensions;
    # We choose batch=3, time=5 for illustration, features=256 is fixed
    batch_size = 3
    time_steps = 5
    features = 256
    return tf.random.uniform(shape=[batch_size, time_steps, features], dtype=tf.float32)

