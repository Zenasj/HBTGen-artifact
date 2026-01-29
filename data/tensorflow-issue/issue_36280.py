# tf.random.uniform((B, 1, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # GRU layer with 1 unit as in the original example
        self.gru = tf.keras.layers.GRU(1, name='my-output')

    def call(self, inputs):
        # Create mask tensor by comparing inputs to zero (not_equal)
        # tf.keras.backend.not_equal returns a bool tensor for masking
        mask = tf.keras.backend.not_equal(inputs, 0)
        # Pass inputs and mask to the GRU layer
        output = self.gru(inputs, mask=mask)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching expected shape: (batch_size, timesteps=1, features=1)
    # Use dtype float32 as matching original example dtype
    batch_size = 4  # arbitrary batch size for example
    return tf.random.uniform((batch_size, 1, 1), dtype=tf.float32)

