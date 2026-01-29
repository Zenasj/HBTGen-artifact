# tf.random.uniform((BATCH_SIZE, seq_length), dtype=tf.int32)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 rnn_units,
                 batch_size,
                 dtype=tf.float32):
        super(MyModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            batch_input_shape=[batch_size, None],
            dtype=dtype)

        self.gru = tf.keras.layers.GRU(
            rnn_units,
            return_sequences=True,
            stateful=True,
            recurrent_initializer='glorot_uniform',
            dtype=dtype)

        self.dense = tf.keras.layers.Dense(vocab_size, dtype=dtype)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.gru(x)
        x = self.dense(x)
        return x


def my_model_function():
    # Hardcoded parameters matching TF text generation example for Shakespeare dataset
    vocab_size = 65       # Unique characters in dataset
    embedding_dim = 256
    rnn_units = 1024
    batch_size = 64       # Typical batch size used in training

    model = MyModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=batch_size)

    # Compile with Adam optimizer and a custom sparse categorical cross entropy loss function
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)

    return model


def GetInput():
    # Return a random int32 tensor of shape (batch_size, seq_length)
    # Here batch_size and seq_length chosen to align with model expectations
    BATCH_SIZE = 64
    SEQ_LENGTH = 100
    VOCAB_SIZE = 65  # must match model vocab_size

    # Create random integers in vocab range [0, VOCAB_SIZE)
    random_input = tf.random.uniform(
        shape=(BATCH_SIZE, SEQ_LENGTH),
        minval=0,
        maxval=VOCAB_SIZE,
        dtype=tf.int32)
    return random_input

