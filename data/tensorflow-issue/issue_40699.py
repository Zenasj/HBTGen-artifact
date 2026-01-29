# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← no explicit shape given, assume (1, 28, 28, 3) as a common conv input example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, filter_nbr=8):
        """Initialize MyModel with a Conv2D and BatchNormalization layer."""
        super(MyModel, self).__init__()
        self.conv11 = tf.keras.layers.Conv2D(filter_nbr, kernel_size=3, padding='SAME')
        self.bn11 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        """Forward pass applying Conv2D, BatchNorm, and ReLU."""
        x = self.conv11(inputs)
        x = self.bn11(x)
        x = tf.nn.relu(x)
        return x

def my_model_function():
    # Return an instance of MyModel. No pretrained weights are included,
    # matching the original example scenario.
    return MyModel()

def GetInput():
    # Return a random tensor input that matches expected input.
    # Since the original issue code does not specify input shape,
    # we assume a generic image batch of size 1, 28x28 pixels, 3 channels.
    # This is common for convolutional models.
    return tf.random.uniform((1, 28, 28, 3), dtype=tf.float32)

