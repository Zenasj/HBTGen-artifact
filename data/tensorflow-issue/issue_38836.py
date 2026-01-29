# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)

import tensorflow as tf
import tensorflow_probability as tfp

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        super().__init__()
        # Using flipout layers from TensorFlow Probability for Bayesian inference as in the issue
        self.conv_flipout = tfp.layers.Convolution2DFlipout(
            6, kernel_size=3, padding="SAME", activation=tf.nn.relu, input_shape=input_shape
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense_flipout = tfp.layers.DenseFlipout(num_classes)

    def call(self, inputs, training=False):
        x = self.conv_flipout(inputs)
        x = self.flatten(x)
        logits = self.dense_flipout(x)
        return logits

def my_model_function():
    # Instantiate MyModel with default MNIST input shape and 10 classes
    return MyModel()

def GetInput():
    # Return a random tensor input matching the MNIST shape (batch_size=64, 28, 28, 1)
    # Using float32 dtype as expected by the model
    batch_size = 64
    img_rows, img_cols, channels = 28, 28, 1
    return tf.random.uniform((batch_size, img_rows, img_cols, channels), dtype=tf.float32)

