# tf.random.uniform((BATCH_SIZE, 200), dtype=tf.int32)
import tensorflow as tf
import numpy as np

# The original model uses two inputs of shape (200,), corresponding to padded tokenized sequences.
# The embedding dimension is 300. The model encodes each input with shared embedding + Conv1D + pooling + flatten layers.
# Then concatenates encoded representations and passes through dense layers to output a sigmoid prediction.

MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 300
NB_WORDS = 200000  # max vocab size, inferred from preprocessing

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer: with random initialization as placeholder since embedding_matrix is not included here.
        # To keep the model self-contained, we initialize embeddings randomly.
        # In practice, pretrained embeddings can be loaded.
        self.embedding = tf.keras.layers.Embedding(
            input_dim=NB_WORDS + 1,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_SEQUENCE_LENGTH,
            # weights optional in original: omitted here
            trainable=True
        )
        self.conv1d = tf.keras.layers.Conv1D(128, 3, activation='tanh')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.maxpool = tf.keras.layers.MaxPooling1D(3)
        self.flatten = tf.keras.layers.Flatten()

        # Dense layers after concatenating encoded sequences
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.prediction = tf.keras.layers.Dense(1, activation='sigmoid')

    def encode(self, x):
        # Encode input sequence to vector representation
        x = self.embedding(x)
        x = self.conv1d(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        return x

    def call(self, inputs, training=False):
        # inputs is a list or tuple of two tensors: left and right sequences
        left, right = inputs
        encoded_left = self.encode(left)
        encoded_right = self.encode(right)

        merged = tf.keras.layers.concatenate([encoded_left, encoded_right])
        x = self.dense1(merged)
        x = self.dense2(x)
        x = self.dense3(x)
        out = self.prediction(x)
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tuple compatible with MyModel input
    # We generate two random integer tensors simulating tokenized padded sequences
    BATCH_SIZE = 16  # example batch size
    input_left = tf.random.uniform(
        (BATCH_SIZE, MAX_SEQUENCE_LENGTH), minval=0, maxval=NB_WORDS, dtype=tf.int32
    )
    input_right = tf.random.uniform(
        (BATCH_SIZE, MAX_SEQUENCE_LENGTH), minval=0, maxval=NB_WORDS, dtype=tf.int32
    )
    return (input_left, input_right)

