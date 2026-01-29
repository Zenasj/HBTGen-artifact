# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from Fashion MNIST images with added channel dimension

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        regularizer = tf.keras.regularizers.L2(1e-5)
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu", kernel_regularizer=regularizer)
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation="relu", kernel_regularizer=regularizer)
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer)
        self.dense2 = tf.keras.layers.Dense(10, kernel_regularizer=regularizer)  # Outputs logits for 10 classes

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return a new instance of MyModel
    return MyModel()

def GetInput():
    # Return a random batch of images with shape [batch_size, 28, 28, 1]
    # batch_size chosen as 128 to simulate a realistic batch
    batch_size = 128
    # Values normalized between 0 and 1, matching dataset preprocessing
    return tf.random.uniform((batch_size, 28, 28, 1), minval=0, maxval=1, dtype=tf.float32)

