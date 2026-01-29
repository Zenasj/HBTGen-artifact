# tf.random.uniform((B,)) ← Input is a 1D tensor since the example model input shape is (1,)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example dummy_model with a single Dense layer, input_shape=(1,)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # Forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Based on the example, input shape is (batch_size, 1)
    # Batch size is arbitrary; here we use 4 as in the example
    batch_size = 4
    # Shape is (batch_size, 1)
    # Use uniform random floats as placeholder input, dtype float32
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)

