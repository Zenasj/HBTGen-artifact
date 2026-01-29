# tf.random.uniform((batch_size, None, 8), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    """
    This model replicates the example from the issue discussion:
    An LSTM-based model processing sequences of variable length and feature dimension 8,
    outputting a scalar regression value per sequence.

    Assumptions:
    - Input shape is (batch_size, sequence_length, 8), sequence_length is variable.
    - This model encapsulates the LSTM + Dense layers from the minimal reproducible example.
    - The purpose is to illustrate a model that can be used in a multi-GPU strategy context.
    """
    def __init__(self):
        super().__init__()
        self.lstm = layers.LSTM(16)
        self.dense = layers.Dense(1)

    def call(self, inputs, training=False):
        """
        Forward pass:
        inputs: tf.Tensor of shape (batch_size, seq_len, 8)
        returns: tf.Tensor of shape (batch_size, 1)
        """
        x = self.lstm(inputs)
        x = self.dense(x)
        return x


def my_model_function():
    """
    Creates and returns an instance of MyModel.
    """
    return MyModel()


def GetInput():
    """
    Returns a sample random tf.Tensor input matching the model input shape:
    A batch of sequences of floats with variable length and feature dimension=8.

    To enable tf.data pipelines with bucket_by_sequence_length etc., shape is (batch_size,seq_len,8).
    Here, we fix batch_size=4 and seq_len=10 arbitrarily as a representative sample.
    """
    batch_size = 4
    seq_len = 10  # fixed length for simplicity in this input generator
    feature_dim = 8
    # Create random uniform input data with float32 dtype
    return tf.random.uniform((batch_size, seq_len, feature_dim), dtype=tf.float32)

