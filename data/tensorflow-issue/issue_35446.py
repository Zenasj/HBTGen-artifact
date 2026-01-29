# tf.random.uniform((B, 10, 128), dtype=tf.float32), tf.random.uniform((B,), dtype=tf.float32)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.seq_len = 10  # fixed sequence length as per example
        self.fea_size = 128
        self.n_class = 5

        name = "test-model"
        self.forward_layer = tf.keras.layers.LSTM(
            units=self.fea_size, return_sequences=True, name=name + "-forward"
        )
        self.backward_layer = tf.keras.layers.LSTM(
            units=self.fea_size, return_sequences=True, go_backwards=True, name=name + "-backward"
        )
        self.bi_lstm = tf.keras.layers.Bidirectional(
            self.forward_layer, backward_layer=self.backward_layer, name=name + "-bi-lstm"
        )
        self.fc = tf.keras.layers.Dense(units=self.n_class, activation="softmax", name="fc-pred")

    def call(self, inputs, training=False):
        # inputs is a tuple/list of two tensors:
        #   input_fea: shape (batch, seq_len=10, fea_size=128)
        #   input_seq: shape (batch,) scalar int32/float32, sequence length to mask
        input_fea, input_seq = inputs

        # Create mask tensor from input_seq:
        # input_seq is scalar per batch: e.g. [10,10,10,...] or shorter sequences
        # tf.sequence_mask expects int32 length
        mask = tf.sequence_mask(tf.cast(input_seq, tf.int32), maxlen=self.seq_len)

        # Apply bidirectional LSTM with mask:
        y = self.bi_lstm(input_fea, mask=mask, training=training)

        # Classify each timestep output:
        y = self.fc(y)

        return y


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return valid random inputs compatible with MyModel:
    # Two inputs:
    # 1) tf.Tensor of shape (batch, 10, 128), dtype float32
    # 2) tf.Tensor of shape (batch,), dtype float32 (sequence length for mask)
    batch_size = 5  # chosen arbitrarily

    input_fea = tf.random.uniform(
        shape=(batch_size, 10, 128), minval=-1.0, maxval=1.0, dtype=tf.float32
    )
    # Using 10 (max seq length) for all batch entries in input_seq
    input_seq = tf.fill(dims=(batch_size,), value=10.0)

    return (input_fea, input_seq)

