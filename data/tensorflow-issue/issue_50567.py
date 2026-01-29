# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← The example input shape is (batch_size, 48, 48, 3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Weight shape corresponds to kernel shape for conv_transpose: [filter_height, filter_width, output_channels, in_channels]
        # Using the shape from the original code: [12, 12, 3, 3]
        self.f = self.add_weight(
            name='kernel',
            shape=[12, 12, 3, 3],
            initializer=tf.random_uniform_initializer(),
            trainable=True,
        )

    def call(self, inp):
        # Here inp shape is (batch_size, 48, 48, 3)
        # The original issue had an error using -1 for batch dimension in output shape on TPU.
        # So we extract batch size explicitly and use static shape for output.
        N = tf.shape(inp)[0]  # dynamic batch size
        H = tf.shape(inp)[1]
        W = tf.shape(inp)[2]
        C = tf.shape(inp)[3]  # Channels

        # Output shape for conv_transpose is [batch_size, H*4, W*4, C]
        # Upscaling factor is 4 as per original code
        output_shape = tf.stack([N, H * 4, W * 4, C])

        # Strides for conv_transpose
        strides = (4, 4)

        # Use tf.nn.conv_transpose with static output shape for TPU compatibility
        y = tf.nn.conv_transpose(inp, self.f, output_shape, strides, padding='SAME')
        return y

def my_model_function():
    # Return an instance of MyModel with initialized weights
    return MyModel()

def GetInput():
    # Return a random float32 tensor simulating a batch of 4 images with shape (4, 48, 48, 3)
    # A batch size 4 chosen to align with TPU typical usage and the printed batch in original logs
    return tf.random.uniform((4, 48, 48, 3), dtype=tf.float32)

