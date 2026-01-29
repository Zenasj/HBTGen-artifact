# tf.random.uniform((10, 16, 16, 3), dtype=tf.float32) ← Inferred input shape from the example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize convolutional layers as in the issue example
        self.l1 = tf.keras.layers.Conv2D(10, 3)
        self.o1 = tf.keras.layers.Conv2D(2, 1, name='o1')  # name provided for output 1
        self.o2 = tf.keras.layers.Conv2D(3, 1, name='o2')  # name provided for output 2

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        y = self.l1(inputs)
        y1 = self.o1(y)  # Output 1 conv
        y1 = tf.reshape(y1, [batch_size, -1, 2])  # Reshape to (batch, time_steps, channels)
        y2 = self.o2(y)  # Output 2 conv
        y2 = tf.reshape(y2, [batch_size, -1, 3])  # Reshape to (batch, time_steps, channels)
        return y1, y2


class Loss1(tf.keras.losses.Loss):
    def call(self, targets, predictions):
        # Compute timestep-wise absolute error, sum over last axis (channels)
        losses = tf.math.abs(predictions - targets)
        # Return per timestep losses without reducing batch dimension,
        # shape: (batch_size, time_steps)
        return tf.reduce_sum(losses, axis=-1)


def my_model_function():
    """
    Returns an instance of MyModel as defined above.
    This model is compatible with timestep-wise sample weighting for multiple outputs.
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor to be passed to MyModel.
    Matches shape used in the example: batch size 10, 16x16 RGB images.
    """
    return tf.random.uniform((10, 16, 16, 3), dtype=tf.float32)

