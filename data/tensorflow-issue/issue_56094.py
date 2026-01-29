# tf.random.uniform((32, 28, 28, 1), dtype=tf.float32) ← input shape as batch=32, height=28, width=28, channel=1 for MNIST data

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Corresponds to the original Sequential model architecture:
        # Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(28,28,1), batch_size=32)
        self.conv2d = layers.Conv2D(
            32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='valid', input_shape=(28, 28, 1)
        )
        self.maxpool2d = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(20, activation='relu')
        self.dense2 = layers.Dense(10, activation='sigmoid')
        
        # Note: The original code used sigmoid final activation instead of softmax.

    def call(self, inputs, training=False):
        """
        Forward pass replicating the described model.
        - The model expects inputs of shape (batch_size, 28, 28, 1).
        - Output shape will be (batch_size, 10).
        """
        x = self.conv2d(inputs)
        x = self.maxpool2d(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return a fresh instance of MyModel,
    # no pretrained weights included due to lack of saved weights context.
    return MyModel()

def GetInput():
    # Generate a valid input tensor matching the expected input shape:
    # Batch size = 32, Height=28, Width=28, Channels=1.
    # Input dtype float32 scaled to [0,1], similar to normalized MNIST images.
    return tf.random.uniform(shape=(32, 28, 28, 1), minval=0, maxval=1, dtype=tf.float32)

