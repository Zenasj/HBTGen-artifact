# tf.random.uniform((1, 2), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 2], dtype=tf.float32)])
    def call(self, x):
        # Custom stable softplus implementation mimicking TensorFlow's kernel logic:
        # For large positive x, softplus(x) ~ x to avoid exp overflow.
        # For large negative x, softplus(x) ~ exp(x).
        # Otherwise, softplus(x) = log(exp(x) + 1) (using log1p for precision).
        threshold = tf.math.log(tf.experimental.numpy.finfo(tf.float32).eps) + 2.0  # approx -17.6
        too_large = x > -threshold  # roughly x > ~17.6 means output = x
        too_small = x < threshold   # roughly x < ~-17.6 means output = exp(x)
        x_exp = tf.exp(x)
        # Use tf.where to select the stable branches:
        # if too_large: output = x
        # else if too_small: output = exp(x)
        # else output = log1p(exp(x))
        result = tf.where(too_large,
                         x,
                         tf.where(too_small,
                                  x_exp,
                                  tf.math.log1p(x_exp)))
        return result

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random tensor in shape (1, 2) to test the model.
    # Values can be large to test stability (e.g. some above 20).
    # We use uniform from [-25, 25] to cover low, mid, high ranges.
    return tf.random.uniform((1, 2), minval=-25.0, maxval=25.0, dtype=tf.float32)

