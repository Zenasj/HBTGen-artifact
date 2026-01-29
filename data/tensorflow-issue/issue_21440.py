# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape is dynamic (None, None) spatial dims, 3 channels RGB image, batch size 1 assumed

import tensorflow as tf

# This code represents a conceptual fuse of the Fully Convolutional Network (FCN) style model 
# that supports variable input spatial dimensions (height and width) with 3 channels.
# The code is adapted to TF 2.x style, assumes input tensors with shape (1, H, W, 3),
# where H and W can be dynamic (None).
#
# The problem discussed in the issue is about converting such flexible input models to TFLite,
# and the difficulty setting None (variable) shape in TensorFlow Lite conversion.
#
# Here, we implement a sample FCN-like MyModel that accepts variable size inputs.
# We provide also a custom fused MyModel that can be reasoned as comparing two networks' outputs.
#
# Since the original issue does not provide explicit model code, the below is an inferred 
# minimal fully convolutional model capable of variable input shape with 3 channel input.
# The GetInput() function produces a sample input tensor with dynamic (randomized) spatial size.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Example encoder block: Conv2D -> BatchNorm -> ReLU
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()

        # Another conv block
        self.conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()

        # Final conv layer outputting segmentation map (e.g., 21 classes for semantic segmentation)
        # Using "same" padding so output has same spatial resolution as input
        self.conv_out = tf.keras.layers.Conv2D(filters=21, kernel_size=1, padding='same', activation=None)

    def call(self, inputs, training=False):
        """
        Forward pass supporting variable input spatial dimensions.
        Assumes inputs shape: (batch_size, H, W, 3)
        """

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        # Output logits for each of 21 classes at each spatial location
        x = self.conv_out(x)

        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input that matches expected input shape:
    # batch size 1; variable height and width; 3 channels (RGB)
    #
    # For simulation, choose random height and width between 128 and 512.
    # dtype float32 as typical for image inputs.

    batch_size = 1
    height = tf.random.uniform(shape=[], minval=128, maxval=512, dtype=tf.int32)
    width = tf.random.uniform(shape=[], minval=128, maxval=512, dtype=tf.int32)
    channels = 3

    # Generate random float input in range [0,1)
    input_tensor = tf.random.uniform(shape=(batch_size, height, width, channels), dtype=tf.float32)

    return input_tensor

