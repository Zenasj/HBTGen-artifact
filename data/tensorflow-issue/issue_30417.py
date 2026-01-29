# tf.random.uniform((B, 16), dtype=...) ← Input shape inferred as (B, 16), matching the keras Input(shape=(16,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a weight variable of shape (16,16)
        # Using add_weight to be consistent with the original example
        self.w = self.add_weight(
            shape=(16, 16),
            initializer="random_normal",
            trainable=True,
            dtype=tf.float32  # Using float32 dtype for compatibility with mixed precision
        )

    def call(self, inputs, **kwargs):
        # The original issue was that slicing a mixed-precision AutoCastVariable caused an error.
        # Workaround: cast variable to tensor before slicing to enable slicing.
        w_tensor = tf.convert_to_tensor(self.w)
        # Slice the first 16 rows and columns (full slice here)
        sliced_w = w_tensor[:16, :16]
        # Compute dot product of inputs and sliced_w
        output = tf.linalg.matmul(inputs, sliced_w)
        return output

def my_model_function():
    # Return an instance of MyModel as requested
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel:
    # Shape (batch_size, 16), batch size can be a small positive integer like 4
    # Using float32 dtype for compatibility with model
    batch_size = 4
    return tf.random.uniform((batch_size, 16), dtype=tf.float32)

