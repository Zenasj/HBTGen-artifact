# tf.random.uniform((B, 2), dtype=tf.float32) ← Input shape inferred as (batch_size, 2)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No learnable parameters in this model; it just computes y=x^2 and its batch_jacobian dy/dx
        # We use the public GradientTape.batch_jacobian to avoid the bug described
        
    @tf.function(jit_compile=True)
    def call(self, x):
        # Computes y = x^2 and dydx = batch jacobian of y wrt x
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = x ** 2
        dydx = tape.batch_jacobian(y, x)
        return y, dydx


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random input tensor matching shape (batch_size, 2), dtype float32
    
    # Use batch size 3 as per example in the issue for demonstration
    batch_size = 3
    input_shape = (batch_size, 2)
    input_tensor = tf.random.uniform(input_shape, dtype=tf.float32)
    return input_tensor

