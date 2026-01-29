# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← Input shape inferred from CIFAR-10 dataset used in example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreating the CNN from the CIFAR-10 TensorFlow tutorial
        # to reflect the code in the issue, which triggered the conv2d error.
        # This model uses Conv2D, MaxPooling2D, Flatten, Dense layers.
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # CIFAR-10 has 10 classes

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Returns an instance of MyModel.
    # The original code used no pretrained weights, so initialized randomly.
    return MyModel()

def GetInput():
    # Generate a random input tensor simulating CIFAR-10 images:
    # shape [batch, height, width, channels] with standard image size 32x32 RGB
    # Use float32 type as typical for model input after normalization
    batch_size = 64  # Batch size commonly used (from issue code)
    input_tensor = tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float32, minval=0.0, maxval=1.0)
    return input_tensor

