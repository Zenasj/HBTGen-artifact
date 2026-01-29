# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred as (batch_size, 1) from the example code using a 1D input tensor.

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # In this scenario, the original issue involves 3 TFLite ops: log, pow, and div.
        # We'll implement internal submodules for these 3 ops for demonstration.
        # Since these are simple element-wise ops, we use Lambda layers.

        self.log_op = tf.keras.layers.Lambda(lambda x: tf.math.log(x), name="log_op")
        self.pow_op = tf.keras.layers.Lambda(lambda x: tf.math.pow(x, 3), name="pow_op")
        self.div_op = tf.keras.layers.Lambda(lambda x: tf.math.divide(x, x), name="div_op")
        # The input shape is (?, 1) according to the example.

    def call(self, inputs):
        # The original example applies ops separately to the same input tensor.
        # To fuse the behavior, we compute all three outputs and compare them somehow.
        # However, since input / input = 1 and input ** 3 is distinct, a direct equality comparison won't be meaningful.
        # Instead, let's produce a dictionary output of all 3 results concatenated along last axis.
        # Alternatively, we can stack along last axis to output a tensor of shape (batch, 3).

        log_res = self.log_op(inputs)
        pow_res = self.pow_op(inputs)
        div_res = self.div_op(inputs)

        # Concatenate results along last dimension for easier output:
        # Shape will be (batch_size, 1, 3) after stacking and then squeezed to (batch_size, 3)
        combined = tf.concat([log_res, pow_res, div_res], axis=-1)
        # For completeness, add a small epsilon to inputs inside log to avoid NaN on zero or negative inputs.
        # But since GetInput generates positive values from 1 upwards, it's safe.

        return combined

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Return a batch of positive input values shaped (batch_size, 1)
    # We use tf.random.uniform with minval=1.0 to avoid log(0) or negative.
    batch_size = 10  # arbitrary batch size for demonstration
    input_tensor = tf.random.uniform((batch_size, 1), minval=1.0, maxval=10.0, dtype=tf.float32)
    return input_tensor

