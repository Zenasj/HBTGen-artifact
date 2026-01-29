# tf.random.uniform((B, 28, 28), dtype=tf.float32)  # Assuming MNIST-like input shape (batch, height, width)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple MNIST-style model from the issue reconstructed here:
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits layer

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits


def my_model_function():
    # Return an instance of MyModel
    # The model should be compiled similarly to the example in the issue if needed (not requested explicitly here)
    return MyModel()


def GetInput():
    # Generate random input matching the assumed input expected by MyModel:
    # MNIST images are 28x28 grayscale, batch size is chosen arbitrarily as 32
    batch_size = 32
    height = 28
    width = 28
    # dtype float32 matches typical TensorFlow input for MNIST
    return tf.random.uniform((batch_size, height, width), dtype=tf.float32)

