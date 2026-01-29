# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST grayscale images of size 28x28

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicates the minimal Conv2D - MaxPool - Dense architecture from the issue
        # Conv2D with 32 filters, kernel size 3x3, "same" padding, ReLU activation
        self.conv = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.maxpool = tf.keras.layers.MaxPool2D(2)
        self.flatten = tf.keras.layers.Flatten()
        # Output Dense layer with 10 units (number of classes), softmax activation
        self.dense = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.maxpool(x)
        x = self.flatten(x)
        output = self.dense(x)
        return output

def my_model_function():
    """
    Factory function that returns an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Generates a random tensor input with the shape and dtype that the model expects.
    Based on the MNIST data shape: batch size is arbitrary (here 32),
    image size: 28x28, single channel grayscale, dtype float32.
    
    This is consistent with the original issue where input shape was (batch, 28, 28, 1).
    """
    batch_size = 32  # Typical batch size; can be changed as needed
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

