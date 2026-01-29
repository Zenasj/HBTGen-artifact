# tf.random.uniform((1, 512, 512, 3), dtype=tf.float32)
import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self, kernel):
        super().__init__()
        # Split kernel into two halves along the output channels dimension
        depth = kernel.shape[-1]
        half_depth = depth // 2
        self.kernel = tf.constant(kernel, dtype=tf.float32)
        self.kernel_1 = tf.constant(kernel[:, :, :, :half_depth], dtype=tf.float32)
        self.kernel_2 = tf.constant(kernel[:, :, :, half_depth:], dtype=tf.float32)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1, 512, 512, 3), dtype=tf.float32)
    ])
    def call(self, x):
        # Single conv2d with full kernel
        out_1 = tf.nn.conv2d(
            x,
            self.kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC",
            dilations=[1, 1],
            name="conv2d_single")

        # Two conv2d with half kernels then concatenated on channels axis
        out_2_part1 = tf.nn.conv2d(
            x,
            self.kernel_1,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC",
            dilations=[1, 1],
            name="conv2d_split_1")

        out_2_part2 = tf.nn.conv2d(
            x,
            self.kernel_2,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC",
            dilations=[1, 1],
            name="conv2d_split_2")

        out_2 = tf.concat([out_2_part1, out_2_part2], axis=3, name="concat_split")

        # Compare the two outputs element-wise close within strict tolerance
        # Using tf.reduce_all with tf.abs(out_1 - out_2) < tolerance
        # The issue states results differ at ~1e-7 scale with rounding differences,
        # so we define a small tolerance; here we use 1e-6 to tolerate numerical noise.
        tolerance = 1e-6
        are_close = tf.reduce_all(tf.abs(out_1 - out_2) <= tolerance)

        # To reflect the difference numerically: output max abs difference.
        max_abs_diff = tf.reduce_max(tf.abs(out_1 - out_2))

        # Return a dictionary of three things:
        # 1) out_1: single conv2d output
        # 2) out_2: concatenated split conv2d output
        # 3) dictionary with 'are_close' bool scalar and max_abs_diff float scalar
        # A user can choose what to use; the presence of differences explains the bug.
        return {
            "conv2d_single": out_1, 
            "conv2d_split_concat": out_2,
            "comparison": {
                "are_close": are_close,
                "max_absolute_difference": max_abs_diff
            }
        }


def my_model_function():
    # Here we generate a sample kernel of depth 16 as default, 
    # as the issue revolves around kernel depth splitting 16 -> 8 + 8.
    import numpy as np
    depth = 16
    # Kernel shape: (3, 3, in_channels=3, out_channels=depth)
    kernel = (np.random.uniform(
        low=-0.05,
        high=0.05,
        size=(3, 3, 3, depth))
    .astype(np.float16)
    .astype(np.float32))
    return MyModel(kernel)


def GetInput():
    # Return a random input tensor matching expected shape (1, 512, 512, 3)
    # The issue uses inputs with values around 2.0 and then shifted:
    import numpy as np
    x = 2 * np.random.rand(1, 512, 512, 3).astype(np.float32)
    input_1_feed = np.where(x > 1, x - 0.5, x - 1.5).astype(np.float32)
    return tf.constant(input_1_feed)

