# tf.random.uniform((10, 9), dtype=tf.bfloat16)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable params, just wrapping raw_ops Cos and Asin operations

    def call(self, x):
        # Apply Cos then Asin using raw_ops as per the reported issue
        x_cos = tf.raw_ops.Cos(x=x)
        x_asin = tf.raw_ops.Asin(x=x_cos)
        return x_asin

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input
    # Shape: [10,9], dtype: bfloat16 (matching original repro)
    # Note: The issue arises due to precision differences in bfloat16 with jit_compile=True
    return tf.random.uniform(shape=(10, 9), dtype=tf.bfloat16)

