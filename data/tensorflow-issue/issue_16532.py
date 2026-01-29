# tf.random.uniform((32, 400), dtype=tf.int32) ← inferred input shape and type from imdb dataset (batch_size=32, sequence length=400)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Parameters inferred from original example
        self.max_features = 20000  # vocab size
        self.embedding_dims = 50
        self.maxlen = 400

        # Embedding layer that maps vocab indices to embedding vectors
        self.embedding = layers.Embedding(input_dim=self.max_features, output_dim=self.embedding_dims, input_length=self.maxlen)

        # Global average pooling over sequence length dimension
        self.global_avg_pool = layers.GlobalAveragePooling1D()

        # Final classification layer with sigmoid output for binary classification
        self.output_dense = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        """
        inputs: a batch of sequences of shape (batch_size, maxlen)
        """
        x = self.embedding(inputs)
        x = self.global_avg_pool(x)
        x = self.output_dense(x)
        return x


def my_model_function():
    """
    Returns an instance of the MyModel class.
    """
    return MyModel()


def GetInput():
    """
    Returns a random integer tensor of shape (batch_size=32, sequence_length=400),
    that represents a batch of token indices appropriate for the MyModel input.
    Values are in range [0, max_features-1].
    """
    batch_size = 32
    maxlen = 400
    max_features = 20000
    return tf.random.uniform(shape=(batch_size, maxlen), minval=0, maxval=max_features, dtype=tf.int32)

