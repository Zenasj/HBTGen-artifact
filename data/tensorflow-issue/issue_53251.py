# tf.random.uniform((1, 2, 4, 4, 3), dtype=tf.float32) ← inferred input shape based on example Conv3D input_shape=(2,4,4,3) with batch_size=1

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model implements a depth-wise 3D convolution using Conv3D with groups equal to input channels.
    Since tf.keras.layers.DepthwiseConv3D does not exist, this is a workaround using Conv3D with groups.
    The model takes input shape (batch_size=1, depth=2, height=4, width=4, channels=3).
    It performs grouped Conv3D with groups=channels (=3), filters=3 (one filter per channel),
    kernel_size=(2,4,4). The output shape will be (1, 1, 1, 1, 3).
    """

    def __init__(self):
        super().__init__()
        # Using Conv3D with groups=input channels to simulate depthwise conv3d
        # kernel size (2,4,4) covers entire spatial dims except channels and batch
        self.depthwise_conv3d = tf.keras.layers.Conv3D(
            filters=3,
            kernel_size=(2, 4, 4),
            groups=3,
            padding='valid',
            use_bias=True,
            kernel_initializer='glorot_uniform'
        )

    def call(self, inputs):
        # Forward pass through depthwise Conv3D
        return self.depthwise_conv3d(inputs)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor with shape matching model input: (1, 2, 4, 4, 3), dtype float32
    return tf.random.uniform((1, 2, 4, 4, 3), dtype=tf.float32)

