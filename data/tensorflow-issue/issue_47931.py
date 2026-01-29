# tf.random.uniform((B, N), dtype=tf.float32) ← Input is a 2D float tensor matching TF-IDF vector shape

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, units=64, dropout_rate=0.2, output_units=1, output_activation='sigmoid'):
        super().__init__()
        # Based on the original model defined by a simple MLP with dropout and dense layers.
        # Dropout is kept without input_shape here, expecting input shape to be defined on call/build.
        self.dropout1 = layers.Dropout(rate=dropout_rate)
        self.dense1 = layers.Dense(units=units, activation='relu')
        self.dropout2 = layers.Dropout(rate=dropout_rate)
        self.dense2 = layers.Dense(units=units, activation='relu')
        self.dropout3 = layers.Dropout(rate=dropout_rate)
        self.output_layer = layers.Dense(units=output_units, activation=output_activation)

    def call(self, inputs, training=False):
        # inputs expected: dense Tensor with shape (batch_size, features)
        x = self.dropout1(inputs, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        x = self.dropout3(x, training=training)
        return self.output_layer(x)

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # The original vectorizer transforms texts to a sparse tf-idf vector of shape (batch_size, vocab_size).
    # Assumptions:
    # - Using a small vocab size to keep this simple (e.g. 50 features)
    # - Batch size is 4 as in training example
    # - Input dtype float32 (tf-idf vector)
    batch_size = 4
    vocab_size = 50  # assumed max features selected by SelectKBest / TF-IDF

    # Create a random dense tensor simulating vectorized input (e.g., tf-idf scores)
    # Range [0,1) to simulate tf-idf normalized values
    input_tensor = tf.random.uniform(
        shape=(batch_size, vocab_size), dtype=tf.float32)
    return input_tensor

