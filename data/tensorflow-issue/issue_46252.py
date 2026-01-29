# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape is (batch, 28, 28) grayscale images, e.g., MNIST digits

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers as in the example Sequential model
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel; this model can be compiled and trained directly.
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input: batch of 32 with (28,28) grayscale images
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

