# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← Input shape inferred from autoencoder model's Conv2D input_shape

import tensorflow as tf
import numpy as np
import json
import os

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # Encoder layers
        self.n_hidden_1 = 16
        self.n_hidden_2 = 12
        self.n_hidden_3 = 10
        self.n_hidden_4 = 8
        self.convkernel = (3, 3)
        self.poolkernel = (2, 2)

        # Activation factory mimic — leaky or linear assumed similar
        # We'll use LeakyReLU with alpha=0.01 as default
        self.activation = tf.keras.layers.LeakyReLU(0.01)

        # Encoder
        self.conv1 = tf.keras.layers.Conv2D(
            self.n_hidden_1, self.convkernel, activation='relu', input_shape=(224, 224, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D(self.poolkernel, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(self.n_hidden_2, self.convkernel, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(self.n_hidden_3, self.convkernel, padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(self.poolkernel, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(self.n_hidden_4, self.convkernel, activation='relu', padding='same')
        self.pool3 = tf.keras.layers.MaxPooling2D(self.poolkernel, padding='same', name='bottleneck')

        # Decoder
        self.deconv1 = tf.keras.layers.Conv2D(self.n_hidden_4, self.convkernel, activation='relu', padding='same')
        self.up1 = tf.keras.layers.UpSampling2D(self.poolkernel)
        self.deconv2 = tf.keras.layers.Conv2D(self.n_hidden_3, self.convkernel, padding='same')
        self.deconv3 = tf.keras.layers.Conv2D(self.n_hidden_2, self.convkernel, padding='same')
        self.up2 = tf.keras.layers.UpSampling2D(self.poolkernel)
        self.deconv4 = tf.keras.layers.Conv2D(32, (1, 1))
        self.up3 = tf.keras.layers.UpSampling2D(self.poolkernel)
        self.output_conv = tf.keras.layers.Conv2D(3, self.convkernel, activation='sigmoid', padding='same')

    def call(self, inputs, training=False):
        # Forward pass through encoder
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.pool3(x)

        # Save bottleneck representation (can be used for comparison if needed)
        bottleneck = x

        # Decoder pass
        x = self.deconv1(x)
        x = self.up1(x)
        x = self.deconv2(x)
        x = self.activation(x)
        x = self.deconv3(x)
        x = self.activation(x)
        x = self.up2(x)
        x = self.deconv4(x)
        x = self.activation(x)
        x = self.up3(x)
        reconstructed = self.output_conv(x)

        return reconstructed

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Based on autoencoder model: input shape (B, 224, 224, 3), float32 in [0,1]
    batch_size = 4  # Reasonable batch size for example
    input_tensor = tf.random.uniform(
        (batch_size, 224, 224, 3), minval=0.0, maxval=1.0, dtype=tf.float32)
    return input_tensor

