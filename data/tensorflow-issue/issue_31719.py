# tf.random.uniform((B, 2), dtype=tf.float64) ← Input shape is (batch_size, 2) with float64 dtype

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
    
    def call(self, x):
        # Compute y = x^2
        y = x * x
        
        # Use GradientTape to compute dy/dx per example in batch
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = x * x
        grad = tape.gradient(y, x)
        return grad

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a batch of random inputs with shape (batch_size=4, 2)
    # Float64 dtype to match model expectations
    batch_size = 4
    input_shape = (batch_size, 2)
    return tf.random.uniform(input_shape, dtype=tf.float64)

