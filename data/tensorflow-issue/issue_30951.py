# tf.random.uniform((B, 30), dtype=tf.float32) and tf.random.uniform((B, 10), dtype=tf.bool) ← Input shapes for the two inputs

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(10)
        # The mask input will be cast from bool to float32 and multiplied elementwise to apply masking on the output
        # This replaces the usage of Lambda layer with mask argument, which caused issues in TensorFlow 1.x.
    
    def call(self, inputs, training=False):
        # inputs expected to be a tuple or list: (inp_tensor, mask_tensor)
        inp, mask = inputs  # inp shape: (batch_size, 30), mask shape: (batch_size, 10), dtype mask: bool
        
        x = self.dense(inp)  # shape (batch_size, 10), float32
        mask_f = tf.cast(mask, dtype=x.dtype)  # cast bool mask to float32
        
        # Apply mask by elementwise multiplication: zero out positions in output where mask is False
        x_masked = x * mask_f
        return x_masked

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a batch of inputs with shapes matching the model input
    batch_size = 20  # picked batch size from example in the issue
    
    # Input tensor: float32, shape (batch_size, 30)
    inp = tf.random.uniform((batch_size, 30), dtype=tf.float32)
    
    # Mask tensor: bool, shape (batch_size, 10)
    # Random boolean mask with roughly 50% True/False
    mask = tf.random.uniform((batch_size, 10), minval=0, maxval=2, dtype=tf.int32)
    mask = tf.cast(mask, dtype=tf.bool)
    
    # Return as a tuple/list consistent with model.call input format
    return (inp, mask)

