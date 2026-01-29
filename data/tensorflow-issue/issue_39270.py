# tf.random.uniform((32, 28, 28), dtype=tf.float32) ← Batch size 32, input shape 28x28 grayscale images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple MNIST-like model as in the issue example:
        # Flatten -> Dense(128, relu) -> Dense(10 logits)
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Construct a fresh instance of MyModel with default initialization.
    # No pretrained weights are provided in the issue.
    return MyModel()

def GetInput():
    # Return a random float32 tensor matching the expected input shape (batch_size=32, 28x28 grayscale)
    # Similar to MNIST samples, normalized to [0,1].
    return tf.random.uniform((32, 28, 28), minval=0, maxval=1, dtype=tf.float32)

