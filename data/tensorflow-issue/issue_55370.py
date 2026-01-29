# tf.random.uniform((B, input_dim), dtype=tf.float32) <- Assuming input shape for Dense layer input is (batch_size, input_dim)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, units=512):
        super().__init__()
        # Single Dense layer from the minimal reproducible example
        self.dense = tf.keras.layers.Dense(units)

    def call(self, inputs, training=False):
        # Forward pass just runs the Dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel with default Dense size 512 units
    return MyModel()

def GetInput():
    # Since the example uses tf.keras.Sequential with Dense(512)
    # Dense requires input shape (batch_size, input_dim), input_dim can be arbitrary
    # Choose input_dim=128 as a reasonable arbitrary input dimension
    batch_size = 8
    input_dim = 128
    # Return a random float32 tensor in shape (batch_size, input_dim)
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

