# tf.random.normal((10, 9, 8, 1, 8, 3, 2), dtype=tf.bfloat16)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Alpha value for LeakyRelu matching the issue example
        self.alpha = 9.456766920329814

    @tf.function(jit_compile=True)
    def call(self, x):
        # Applying tf.raw_ops.Cos followed by tf.raw_ops.LeakyRelu as in the issue
        x_cos = tf.raw_ops.Cos(x=x)
        x_leaky = tf.raw_ops.LeakyRelu(features=x_cos, alpha=self.alpha)
        return x_leaky

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Input shape and dtype as provided in the original issue reproduction code:
    # Shape: [10, 9, 8, 1, 8, 3, 2]
    # dtype: bfloat16
    return tf.random.normal([10, 9, 8, 1, 8, 3, 2], dtype=tf.bfloat16)

