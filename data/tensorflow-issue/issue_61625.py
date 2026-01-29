# tf.random.uniform((3, 14, 14, 576), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The problematic ZeroPadding2D layer using very large padding values
        # which caused overflow in the bug report.
        # Using the same large padding values as in the original issue.
        padding_0_0 = 125091515651
        padding_0_1 = 125091515651
        padding_0 = [padding_0_0, padding_0_1]
        padding_1_0 = 125091515651
        padding_1_1 = 125091515651
        padding_1 = [padding_1_0, padding_1_1]
        padding = [padding_0, padding_1]

        # Note: This large padding will cause overflow in actual execution,
        # replicating the bug scenario.
        self.zero_padding = tf.compat.v1.keras.layers.ZeroPadding2D(
            padding=padding,
            data_format=None
        )

    def call(self, inputs):
        # Forward through the ZeroPadding2D layer
        return self.zero_padding(inputs)

def my_model_function():
    # Return an instance of MyModel with the large padding causing overflow
    return MyModel()

def GetInput():
    # Generate a random tensor input of shape (3, 14, 14, 576), dtype float32
    return tf.random.uniform(shape=(3, 14, 14, 576), dtype=tf.float32)

