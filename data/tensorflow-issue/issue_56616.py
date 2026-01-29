# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from Fashion MNIST grayscale images 28x28x1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Building the same sequential layers as in the provided example:
        # Conv2D(64, 3x3) -> MaxPooling2D(2x2) -> Conv2D(64, 3x3) -> MaxPooling2D(2x2)
        # -> Flatten -> Dense(128, relu) -> Dense(10, softmax)
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel matching the architecture described
    return MyModel()

def GetInput():
    # Return a random tensor to simulate input images with batch size B=1,
    # height=28, width=28, channels=1 as normalized float32 values between 0 and 1.
    return tf.random.uniform((1, 28, 28, 1), minval=0, maxval=1, dtype=tf.float32)

