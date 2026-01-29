# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST data (28x28 grayscale images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the Sequential model from the issue:
        # Flatten input (28,28), Dense 128 relu, Dense 10 logits
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits output

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Return an instance of MyModel. No pretrained weights are referenced, create fresh instance.
    return MyModel()

def GetInput():
    # Return a random float32 tensor with shape (B,28,28) normalized between 0 and 1
    # Use batch size 128 as in the example training batch size
    batch_size = 128
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

