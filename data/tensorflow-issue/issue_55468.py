# tf.random.uniform((B, 120), dtype=tf.int32)

import tensorflow as tf

embedding_dim = 16
filters = 128
kernel_size = 5
dense_dim = 6
vocab_size = 10000  # Assumed vocabulary size since not specified
max_length = 120    # Input sequence length inferred from model summary

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length)
        self.conv1d = tf.keras.layers.Conv1D(filters, kernel_size, activation='relu')
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense1 = tf.keras.layers.Dense(dense_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)              # shape: (batch, max_length, embedding_dim)
        x = self.conv1d(x)                      # shape: (batch, max_length - kernel_size + 1, filters)
        x = self.global_max_pool(x)             # shape: (batch, filters)
        x = self.dense1(x)                      # shape: (batch, dense_dim)
        x = self.dense2(x)                      # shape: (batch, 1), sigmoid output for binary classification
        return x

def my_model_function():
    # Returns a compiled instance of MyModel with settings similar to the original.
    model = MyModel()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def GetInput():
    # Return a random integer tensor shaped (batch_size, max_length) matching vocab input
    # Batch size is assumed 32 as reasonable default for testing
    batch_size = 32
    # Input is integers in range [0, vocab_size)
    return tf.random.uniform(shape=(batch_size, max_length), minval=0, maxval=vocab_size, dtype=tf.int32)

