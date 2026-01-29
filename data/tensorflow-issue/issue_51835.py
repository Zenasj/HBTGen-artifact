# tf.random.uniform((B, 32, 32, 3), dtype=...)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This simulates the ResAdd layer from the issue
        # Originally used self.add_weight for res_gain, but this caused saving issues.
        # The fix was to use a tf.Variable instead.
        self.res_gain = tf.Variable(0.0, trainable=True, dtype=tf.float32)

    def call(self, inputs):
        # inputs is expected to be a tuple/list of two tensors (res, skip)
        res, skip = inputs
        gain = tf.cast(self.res_gain, res.dtype)
        out = res * gain + skip
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a dummy input consistent with the issue:
    # Input shape: (batch_size, 32, 32, 3)
    # The model's call expects a tuple of 2 tensors (res, skip) both of this shape.
    # Use batch size 1 as a default minimal batch.
    batch_size = 1
    shape = (batch_size, 32, 32, 3)
    x = tf.random.uniform(shape, dtype=tf.float32)
    return (x, x)

