# tf.random.uniform((B=1, H=256, W=256, C=1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize a 3x3 convolution kernel for single-channel input
        # Shape: (3, 3, in_channels=1, out_channels=1)
        # Inspired by the original code using tf.random_normal([3,3])
        kernel_init = tf.random.normal([3, 3, 1, 1])
        self.kernel = tf.Variable(kernel_init, trainable=True)

    @tf.function
    def call(self, inputs):
        """
        Inputs:
            inputs: 4D tensor with shape [B, H, W, C=1], dtype float32
        Returns:
            conv_out: 4D tensor after convolution with kernel, same shape as inputs
        """

        # Force inputs to have shape [B, H, W, 1] if not already
        # Assume inputs shape: (B, H, W) or (B, H, W, 1)
        x = inputs
        if x.shape.rank == 3:
            # Expand channels dim if missing
            x = tf.expand_dims(x, axis=-1)

        # Perform conv2d with 'SAME' padding, stride 1
        # Explicitly place convolution op on CPU to avoid GPU layout optimizer bug
        with tf.device('/cpu:0'):
            conv_out = tf.nn.conv2d(x, self.kernel, strides=[1,1,1,1], padding='SAME', data_format='NHWC')

        return conv_out

def my_model_function():
    """
    Return an instance of MyModel with initialized trainable kernel.
    """
    return MyModel()

def GetInput():
    """
    Returns a random 4D float32 tensor matching MyModel input shape:
       [batch_size=1, height=256, width=256, channels=1]
    The batch size 1 is chosen for simplicity and compatibility with original example.
    """
    return tf.random.uniform([1, 256, 256, 1], dtype=tf.float32)

