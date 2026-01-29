# tf.random.uniform((B, 3, 3, 10), dtype=tf.float32) ← input shape inferred from Conv2D input layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Convolution layer as in the original example
        self.conv = tf.keras.layers.Conv2D(100, (3, 3))
        # Keras Reshape layer (unfused version)
        self.keras_reshape = tf.keras.layers.Reshape((50, 2))
        # No separate layer needed for tf.reshape, will do inline reshape in call()

    def call(self, inputs, training=False):
        x = self.conv(inputs)

        # Perform two versions of reshape:
        # 1. Using Keras Reshape layer (might not fold in TFLite)
        x_keras_reshaped = self.keras_reshape(x)

        # 2. Using tf.reshape directly (better folding in TFLite)
        # Here, batch size inferred dynamically: (-1, 50, 2)
        x_tf_reshaped = tf.reshape(x, (-1, 50, 2))

        # For demonstration, compare the two reshape outputs
        # They should be identical in values and shape.

        # Compute element-wise equality with some tolerance to handle floating point
        # Using tf.reduce_all and tf.math.abs difference
        is_close = tf.reduce_all(
            tf.math.abs(x_keras_reshaped - x_tf_reshaped) < 1e-6)

        # Return a dict of outputs to observe all tensors,
        # plus the boolean indicating if reshapes are effectively equal.
        return {
            "keras_reshape": x_keras_reshaped,
            "tf_reshape": x_tf_reshaped,
            "equal": is_close
        }

def my_model_function():
    # Return an instance of MyModel with no special initialization needed
    return MyModel()

def GetInput():
    # Return a random input tensor matching input shape of (batch, 3, 3, 10)
    # Use batch size 1 here to reflect the common Keras default shape and to match converter workaround
    # Use float32 dtype consistent with default Conv2D dtype
    return tf.random.uniform((1, 3, 3, 10), dtype=tf.float32)

