# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← CIFAR-10 image input shape

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv block 1
        self.conv1 = layers.Conv2D(32, 3, padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D()

        # Conv block 2
        self.conv2 = layers.Conv2D(64, 3, padding='valid')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D()

        # Conv block 3
        self.conv3 = layers.Conv2D(128, 3, padding='valid')
        self.bn3 = layers.BatchNormalization()

        # Classification head
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(10)  # logits for 10 classes

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # Forward pass with batch normalization and ReLU activations as per original code
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching CIFAR-10 shape (batch size 4 chosen arbitrarily)
    return tf.random.uniform((4, 32, 32, 3), dtype=tf.float32)

