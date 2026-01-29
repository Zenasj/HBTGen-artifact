# tf.random.uniform((1, 512), dtype=tf.int32) ← Input shape inferred from the original tf.keras.Sequential example with batch_size=1, sequence length=512, dtype=int32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer: input_dim=1000 vocabulary size, output_dim=128 embedding size
        self.embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=128)
        # LSTM layer with 64 units, returning sequences (output shape: [batch, time, features])
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=True)
        # Dense layer with 5000 units (likely logits for 5000 classes)
        self.dense = tf.keras.layers.Dense(5000)
        # Argmax layer along the last axis
        self.argmax_layer = tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis=-1))

    def call(self, inputs, training=False):
        """
        Forward pass:
        inputs: int32 tensor of shape (batch_size=1, sequence_length=512)
        output: int64 tensor of shape (batch_size=1, sequence_length=512), each element is argmax index over 5000 classes

        This replicates the original model and its argmax final output.
        """
        x = self.embedding(inputs)       # shape: (1, 512, 128)
        x = self.lstm(x)                 # shape: (1, 512, 64)
        x = self.dense(x)                # shape: (1, 512, 5000)
        out = self.argmax_layer(x)       # shape: (1, 512), int64 indices
        return out

def my_model_function():
    # Return an instance of MyModel.
    # Note: no weights checkpoint loading necessary as the original code snippet did not specify custom weights.
    return MyModel()

def GetInput():
    # Create a random input tensor matching expected input shape and dtype:
    # shape=(1, 512), dtype=tf.int32, values in range [0,999] (embedding vocab size)
    return tf.random.uniform(shape=(1, 512), minval=0, maxval=1000, dtype=tf.int32)

