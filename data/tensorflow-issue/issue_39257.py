# tf.random.uniform((B, 61, 69), dtype=tf.float32) ← input shape inferred from train_tokens_X numpy array shape

import tensorflow as tf
import math

# The original model in the issue is a sequential model with:
# - 3 GRU layers (128 units, no bias, first two return sequences)
# - Flatten
# - Dense with softmax output (output_size=3943)
# The input shape is (None, 61, 69) where 61 is sequence length, 69 is feature size
# Here B=batch size is dynamic.

class MyModel(tf.keras.Model):
    def __init__(self, output_size=3943, max_id=68):
        super().__init__()
        self.gru1 = tf.keras.layers.GRU(128, return_sequences=True, use_bias=False,
                                        input_shape=(None, max_id + 1))
        self.gru2 = tf.keras.layers.GRU(128, return_sequences=True, use_bias=False)
        self.gru3 = tf.keras.layers.GRU(128, use_bias=False)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_size, activation="softmax")

    def call(self, inputs, training=False):
        x = self.gru1(inputs, training=training)
        x = self.gru2(x, training=training)
        x = self.gru3(x, training=training)
        x = self.flatten(x)
        output = self.dense(x)
        return output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor with shape matching model input: (batch_size, 61, 69)
    # Use batch size 256 as in original code
    batch_size = 256
    seq_length = 61
    feature_size = 69
    # Float32 tensor with uniform distribution
    return tf.random.uniform((batch_size, seq_length, feature_size), dtype=tf.float32)

