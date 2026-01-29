# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape is dynamic but typical input is (batch, 256, 256, 1)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the conv layers similar to the original example
        self.conv1 = tf.keras.layers.Conv2D(256, 3, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(256, 3, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(1, 3, padding='same')
        # Identity layer using Lambda
        self.identity = tf.keras.layers.Lambda(lambda x: x)
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.identity(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model with mse loss and the custom PSNR and SSIM metrics adapted from the issue
    def keras_psnr(y_true, y_pred):
        # Dynamic max_pixel range for PSNR as in original code
        max_pixel = tf.math.reduce_max(y_true)
        min_pixel = tf.math.reduce_min(y_true)
        return tf.image.psnr(y_true, y_pred, max_pixel - min_pixel)

    def keras_ssim(y_true, y_pred):
        max_pixel = tf.math.reduce_max(y_true)
        min_pixel = tf.math.reduce_min(y_true)
        return tf.image.ssim(y_true, y_pred, max_pixel - min_pixel)
    
    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=[keras_psnr, keras_ssim]
    )
    return model

def GetInput():
    # Return a random float32 tensor shaped (batch_size, height, width, channels)
    # Matching the data example: batch_size=8, height=256, width=256, channels=1
    batch_size = 8
    height = 256
    width = 256
    channels = 1
    # Use tf.random.uniform to generate values in [0,1)
    return tf.random.uniform(shape=(batch_size, height, width, channels), dtype=tf.float32)

