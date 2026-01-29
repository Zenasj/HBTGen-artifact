# tf.random.uniform((BATCH_SIZE, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST dataset and model input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the same model architecture as in the issue:
        # Input: (28, 28), GRU with 5 units, return sequences, Flatten, Dense(10) output logits
        self.gru = tf.keras.layers.GRU(5, return_sequences=True)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        """
        Forward pass:
        inputs: shape (batch_size, 28, 28), dtype float32, normalized [0,1]

        Output: logits tensor, shape (batch_size, 10)
        """
        x = self.gru(inputs)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    """
    Return an instance of MyModel.
    This mimics the original Sequential model in the issue which was trained on MNIST digits.
    """
    model = MyModel()
    # Note: The original model was trained before saving, weights are not included here since training code is not asked.
    # If pretrained weights required, they should be loaded here.
    return model

def GetInput():
    """
    Generate a random tensor input matching MNIST input shape and type (float32, normalized).
    Shape: (batch_size=1, 28, 28)
    Values: uniform in [0, 1] as per normalization in the example.
    """
    BATCH_SIZE = 1
    H, W = 28, 28
    # Use float32 to match model input dtype
    return tf.random.uniform((BATCH_SIZE, H, W), minval=0, maxval=1, dtype=tf.float32)

