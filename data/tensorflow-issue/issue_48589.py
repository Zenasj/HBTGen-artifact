# tf.random.uniform((1, 1, 1, 1), dtype=tf.float32) ← input shape inferred from example usage in issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Parameters as described in the issue example
        self.filters = 3
        self.kernel_size = (2, 2)
        self.strides = 2
        self.padding = 'valid'

        # Conv2D layer that in invalid conditions produces a zero-dim output instead of raising
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            # kernel initializer is default, no weights pretrained
        )

    def call(self, inputs):
        # Apply Conv2D layer
        output = self.conv(inputs)

        # After conv call, check if output has non-zero spatial dimensions.
        # This mimics what the issue reports: no exception raised on invalid input,
        # but output shape might have zero dimension(s).
        output_shape = tf.shape(output)
        height = output_shape[1]
        width = output_shape[2]

        # If height or width is <= 0, raise a ValueError to mimic expected behavior.
        # Since tf.shape returns a tensor, we use a tf.control_dependencies guard.
        def raise_invalid():
            raise ValueError(
                f"Invalid input shape leads to zero or negative spatial dims after Conv2D: "
                f"height={height.numpy()}, width={width.numpy()}"
            )

        # Because eager mode allows .numpy(), try to check it here.
        if tf.executing_eagerly():
            h = height.numpy()
            w = width.numpy()
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid conv output spatial dimension: height={h}, width={w}")
        else:
            # In graph mode, add an assert op to force the error
            check = tf.debugging.assert_greater(
                height, 0, message="Conv2D output height dimension is zero or negative"
            )
            with tf.control_dependencies([check]):
                output = tf.identity(output)

            check_w = tf.debugging.assert_greater(
                width, 0, message="Conv2D output width dimension is zero or negative"
            )
            with tf.control_dependencies([check_w]):
                output = tf.identity(output)

        return output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Based on the issue example: (batch=1, height=1, width=1, channels=1) float32
    return tf.random.uniform((1, 1, 1, 1), dtype=tf.float32)

