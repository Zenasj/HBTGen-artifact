# tf.random.uniform((B, 13), dtype=tf.float32) ← Assumed batch input with 13 features (mixed int and float encoded as floats)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on build_model() from the provided issue:
        # Input features: 6 integer categorical, 1 string categorical, 6 numerical features
        # Encoders and normalizers used in original code are part of dataset preprocessing.
        # Here we assume input preprocessed and concatenated into a single tensor of shape (B, 13)
        # for inference/testing purpose.

        # We'll create a small dense network mimicking the original:
        self.dense1 = layers.Dense(32, activation="relu")
        self.dropout = layers.Dropout(0.5)
        self.out = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        # inputs shape expected: (B, 13), preprocessed features concatenated.

        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        output = self.out(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # According to the original example, the model inputs are 13 features:
    # 6 categorical encoded as binary (assumed encoded, but for simplicity create float inputs),
    # 1 categorical string encoded,
    # 6 numerical normalized features.
    # Since they are concatenated as one vector, we produce a random float tensor.

    # Assumptions:
    # Batch size = 32 typical for training. Can be flexible.
    batch_size = 32
    feature_dim = 13  # Total features concatenated as in original example

    # Generate random float inputs in range [0,1). In practice, these would be normalized/encoded inputs.
    input_tensor = tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)
    return input_tensor

