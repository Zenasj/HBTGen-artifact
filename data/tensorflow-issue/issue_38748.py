# tf.random.uniform((32, 28, 28), dtype=tf.float32) ← Input shape inferred as MNIST batches with batch size 32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple Dense layer as in original example (input flattened from (28,28))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        # inputs shape: (B, 28, 28), typical MNIST
        x = self.flatten(inputs)
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the MNIST 28x28 "image" shape with batch size 32
    # dtype float32 to match preprocessing in original snippet
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

