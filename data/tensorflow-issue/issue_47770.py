# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST example in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the model given in the issue's snippet
        # A Sequential model with Flatten, Dense(128, relu), Dropout(0.2), Dense(10) layers
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)
    
    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel, no weights to load as this is a freshly defined model matching the example
    return MyModel()

def GetInput():
    # Return a random tensor with batch size 1 of 28x28 grayscale images to match the input shape of the model
    # dtype float32 aligns with typical tf.keras MNIST data pipeline
    batch_size = 1
    height = 28
    width = 28
    # Creating a tensor with values between 0 and 1
    return tf.random.uniform((batch_size, height, width), dtype=tf.float32)

