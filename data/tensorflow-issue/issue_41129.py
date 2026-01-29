# tf.random.uniform((3, 5, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM with 256 units, return sequences
        self.lstm = tf.keras.layers.LSTM(
            units=256,
            return_sequences=True,
            return_state=False)
        # Dense hidden layers as given - two layers of 256 units with relu activation
        self.hidden1 = tf.keras.layers.Dense(256, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(256, activation='relu')
        # Final dense layer to produce output of shape (sequence_length, 1)
        self.out_dense = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=False):
        """
        inputs: tuple of (x_input, mask)
            x_input: tf.Tensor of shape (batch_size, sequence_length, 1), dtype float32
            mask: tf.Tensor of shape (batch_size, sequence_length), dtype bool
        """
        x_input, mask = inputs
        # Pass through LSTM with mask
        x = self.lstm(x_input, mask=mask, training=training)
        x = self.hidden1(x)
        x = self.hidden2(x)
        output = self.out_dense(x)
        return output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Create a random input tensor matching (batch_size=3, sequence_length=5, features=1)
    batch_size = 3
    sequence_length = 5
    feature_dim = 1
    dtype = tf.float32

    x = tf.random.uniform((batch_size, sequence_length, feature_dim), dtype=dtype)
    # Create boolean mask tensor with random valid lengths per batch element
    seq_lengths = tf.random.uniform(
        shape=(batch_size,), minval=0, maxval=sequence_length + 1, dtype=tf.int32)
    mask = tf.sequence_mask(seq_lengths, maxlen=sequence_length)
    return (x, mask)

