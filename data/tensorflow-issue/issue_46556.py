# tf.random.uniform((64, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST dataset and batch_size=64

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple MNIST CNN model replicated from the Estimator example in the issue
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.maxpool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.logits_layer = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        x /= 255.0  # scale inputs (replicating input pipeline scaling)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.logits_layer(x)
        return logits


def my_model_function():
    # Simply return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a batch of random images like MNIST: batch 64, 28x28 grayscale, float32 scaled 0..255
    # The model input expects shape (64, 28, 28, 1) float32 scaled 0..255 originally, here random uniform [0,255)
    return tf.random.uniform((64, 28, 28, 1), minval=0, maxval=255, dtype=tf.float32)

