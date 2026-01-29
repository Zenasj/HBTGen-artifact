# tf.random.uniform((B, (degree,), dtype=tf.float32)) ← Inferred input shape: 1D tensor vector with length 'degree'

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, degree=1):
        super().__init__()
        # Degree corresponds to input vector size
        # Note: The original example used keras.Sequential with Dense layers and input_dim=degree
        self.dense1 = tf.keras.layers.Dense(1, input_shape=(degree,))
        self.dense2 = tf.keras.layers.Dense(1)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Default degree=1 for compatibility with original example; can be changed
    return MyModel(degree=1)

def GetInput():
    # Generate random input tensor matching shape (batch_size, degree)
    # Assuming batch size 1, degree 1 as minimal working example based on provided info
    degree = 1
    batch_size = 1
    return tf.random.uniform((batch_size, degree), dtype=tf.float32)

