# tf.random.uniform((7, 4), dtype=tf.bfloat16)  ← Inferred input shape and dtype from dic['ele']

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Constants used in tf.raw_ops.DivNoNan and tf.raw_ops.Asin calls
        self.divisor = tf.constant(
            [[-921600.0, -499712.0, 638976.0, 344064.0]],
            dtype=tf.bfloat16
        )

    @tf.function(jit_compile=True)
    def call(self, x):
        # Perform element-wise division with no nan output
        x = tf.raw_ops.DivNoNan(y=x, x=self.divisor)
        # Compute arcsine of the division result
        x = tf.raw_ops.Asin(x=x)
        return x

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Generate input tensor to match shape and dtype expected by MyModel
    # Based on the input dic['ele'] from the issue, shape is (7,4) with dtype bfloat16.
    # Use tf.random.uniform with a range compatible with input values
    # The input values in the issue ranged roughly from -1 million to about 1 million, but bfloat16 precision is limited.
    # Using a smaller range since very large inputs divided can cause NaNs or infs.
    shape = (7, 4)
    dtype = tf.bfloat16
    # Generate values roughly in range [-1e6, 1e6] as floats, then cast to bfloat16
    # Match input distribution loosely with scaled uniform
    low = -1e6
    high = 1e6
    x = tf.random.uniform(shape, minval=low, maxval=high, dtype=tf.float32)
    x = tf.cast(x, dtype)
    return x

