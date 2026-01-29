# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define convolutional layers similar to conv_model in the issue
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=5, padding='same', activation='relu', name='conv_layer1')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')
        
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=5, padding='same', activation='relu', name='conv_layer2')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.dense_logits = tf.keras.layers.Dense(10)  # 10 classes for digits 0-9

    def call(self, inputs, training=False):
        # Expect inputs of shape (batch, 28, 28, 1)
        x = self.conv1(inputs)       # (batch, 28, 28, 32)
        x = self.pool1(x)            # (batch, 14, 14, 32)
        
        x = self.conv2(x)            # (batch, 14, 14, 64)
        x = self.pool2(x)            # (batch, 7, 7, 64)
        
        x = self.flatten(x)          # (batch, 7*7*64)
        x = self.dense1(x)           # (batch, 1024)
        x = self.dropout(x, training=training)
        
        logits = self.dense_logits(x) # (batch, 10)
        return logits


def my_model_function():
    # Instantiate and return the model.
    model = MyModel()
    # Build the model by running a dummy input through it once (to create weights)
    dummy_input = tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)
    model(dummy_input, training=False)
    return model


def GetInput():
    # Return a random tensor input matching the expected model input shape
    # Shape: (batch_size, height, width, channels) = (100, 28, 28, 1)
    # Batch size chosen to be 100 matching the batch size in the original script
    return tf.random.uniform((100, 28, 28, 1), dtype=tf.float32)

