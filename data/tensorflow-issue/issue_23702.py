# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST example in the issue (batch size B is variable)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the example model from the issue:
        # Conv2D(8, kernel_size=3, activation='relu', input_shape=(28,28,1))
        self.conv = tf.keras.layers.Conv2D(
            8, kernel_size=3, activation='relu', input_shape=(28, 28, 1))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.flatten(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel.
    # Weights are randomly initialized by default.
    return MyModel()

def GetInput():
    # Return a random input tensor with shape (batch_size, 28, 28, 1).
    # Batch size is chosen as 10 (arbitrary reasonable batch size).
    # Using float32 to match example image dtype.

    batch_size = 10
    return tf.random.uniform(shape=(batch_size, 28, 28, 1), dtype=tf.float32)

