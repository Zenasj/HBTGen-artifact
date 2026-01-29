# tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32) ← input shape inferred from MNIST dataset preprocessing in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10, noise_std=0.7498748441096037):
        super().__init__()
        # GaussianNoise layer as in the example
        self.gaussian_noise = tf.keras.layers.GaussianNoise(stddev=noise_std)
        
        # Flatten layer to flatten the 28x28x1 input to a vector
        self.flatten = tf.keras.layers.Flatten()
        
        # Dense layer with num_classes outputs
        # According to the comments, activation='relu' caused NaNs,
        # so using 'softmax' is the recommended fix for classification.
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs, training=False):
        # During training, GaussianNoise layer will add noise
        x = self.gaussian_noise(inputs, training=training)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Instantiate the model with default parameters:
    # num_classes=10 (for MNIST), noise stddev as per example
    return MyModel()

def GetInput():
    # MNIST input shape: batches of 28x28 grayscale images with float32 values scaled [0,1]
    # Use batch size 110 as in the original code snippet
    batch_size = 110
    img_rows, img_cols, channels = 28, 28, 1
    # Generate a random tensor similar to normalized MNIST images
    # Values between 0 and 1, float32 dtype
    return tf.random.uniform(
        (batch_size, img_rows, img_cols, channels), 
        minval=0.0, maxval=1.0, dtype=tf.float32
    )

