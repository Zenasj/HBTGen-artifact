# tf.random.normal((2, 32, 900, 240), dtype=tf.float32)

import tensorflow as tf

# Hyperparameters inferred from the issue
batch_size = 32
max_out_len = 200
num_hidden = 400
num_classes = 73
max_time_steps = 900
num_features = 240


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # We encapsulate both variants of the model from the issue:
        # 1) LSTMCell with RNN wrapper (slower, higher mem)
        self.forward_cell = [tf.keras.layers.LSTMCell(num_hidden, kernel_initializer='glorot_uniform')]
        self.rnn_from_cells = tf.keras.layers.RNN(
            self.forward_cell, time_major=False, return_sequences=True)

        # 2) Native LSTM layer (faster, less mem, cuDNN optimized)
        self.lstm = tf.keras.layers.LSTM(
            num_hidden, kernel_initializer='glorot_uniform',
            time_major=False, return_sequences=True)

        # Shared dense layer to convert RNN output to logits/classes
        self.dense_for_cells = tf.keras.layers.Dense(
            num_classes, use_bias=True,
            kernel_initializer=tf.initializers.TruncatedNormal(mean=0, stddev=0.1))

        self.dense_for_lstm = tf.keras.layers.Dense(
            num_classes, use_bias=True,
            kernel_initializer=tf.initializers.TruncatedNormal(mean=0, stddev=0.1))

    def call(self, inputs):
        # inputs is a tuple (x, seq_len)
        x, seq_len = inputs

        # Create mask based on sequence lengths for padding
        mask = tf.sequence_mask(seq_len, maxlen=max_time_steps)

        # Output from RNN using LSTMCell wrapper
        state_cells = self.rnn_from_cells(inputs=x, mask=mask)
        logits_cells = self.dense_for_cells(state_cells)

        # Output from native LSTM layer
        state_lstm = self.lstm(inputs=x, mask=mask)
        logits_lstm = self.dense_for_lstm(state_lstm)

        # Compute max absolute difference between the two outputs as a float tensor
        diff = tf.reduce_max(tf.abs(logits_cells - logits_lstm))

        # Return a dictionary with both outputs and the difference for inspection
        # This demonstrates a fusion and comparison of both models as per task requirement
        return {
            "logits_cells": logits_cells,
            "logits_lstm": logits_lstm,
            "max_abs_diff": diff
        }


def my_model_function():
    model = MyModel()
    # Build the model explicitly providing input shapes for TF 2.x compatibility
    # Inputs shape: (batch_size, max_time_steps, num_features), (batch_size,)
    model.build(input_shape=[(batch_size, max_time_steps, num_features),
                             (batch_size,)])
    return model


def GetInput():
    # Generate input tuple (x, seq_len) matching MyModel's expected inputs.
    # As per issue's data:
    # x shape is (2, batch_size, max_time_steps, num_features)
    # seq_len shape is (2, batch_size)
    # We return one "batch" as model expects (batch_size, max_time_steps, num_features)

    # To support batched call like model((x, seq_len)), we'll return inputs matching the first dimension.

    # Using the original example from the issue, but adjusting to single batch,
    # since the model expects (batch_size, time, feat)

    # For demonstration, pick the 0-th "batch" slice which matches model input shape

    # I'll create a random normal tensor with shape (batch_size, max_time_steps, num_features)
    x = tf.random.normal(shape=(batch_size, max_time_steps, num_features), dtype=tf.float32)

    # Sequence lengths: random ints ranging from 1 to max_time_steps for each sample in batch
    # For simplicity, assign a fixed realistic length, e.g., 7 (as in the issue)
    seq_len = tf.constant([7] * batch_size, dtype=tf.int32)

    return (x, seq_len)

