# tf.random.uniform((128, 32, 32, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A Conv2D layer with 32 filters and kernel size 3x3
        self.conv = tf.keras.layers.Conv2D(32, 3)
        # BatchNormalization layer with momentum=0.0 as per the issue
        self.bn = tf.keras.layers.BatchNormalization(axis=3, momentum=0.0)
        # LeakyReLU activation with alpha=0.01
        self.act = tf.keras.layers.LeakyReLU(0.01)

    def call(self, inputs, training):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.act(x)
        return x

def my_model_function():
    # Return fresh instance of the model
    return MyModel()

def GetInput():
    # Input shape is (128, 32, 32, 1) as per example in the issue text.
    # Generate a random normal tensor matching the shape and dtype float32.
    return tf.random.normal((128, 32, 32, 1), dtype=tf.float32)

