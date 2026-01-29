# tf.random.uniform((1, 1, 2, 2, 3), dtype=tf.float32) ← Input shape from example usage in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original problematic ZeroPadding3D with extremely large padding value (causing overflow)
        self.original_zero_padding = tf.keras.layers.ZeroPadding3D(padding=1610612736)
        
        # A safe ZeroPadding3D with moderate padding to demonstrate correct functionality
        self.safe_zero_padding = tf.keras.layers.ZeroPadding3D(padding=1)
    
    def call(self, inputs, training=None):
        # Try to run the original large-padding zero padding, but catch overflow errors internally
        try:
            # This triggers the overflow bug as described in the issue
            original_output = self.original_zero_padding(inputs)
        except Exception as e:
            # Instead of crashing, return an error-indicative tensor (all zeros)
            # and include the error message as a tensor of bytes is non-trivial,
            # so just provide a tensor of zeros with the input shape and a flag.
            original_output = tf.zeros_like(inputs)
        
        # Run safe zero padding as a comparison output
        safe_output = self.safe_zero_padding(inputs)

        # Return both outputs to allow comparison:
        # A tuple with the 'problematic output' (possibly zeros on error)
        # and the safe padding output
        return original_output, safe_output

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Return a random float32 tensor matching the input shape given in the issue
    # Shape: (batch=1, depth=1, height=2, width=2, channels=3)
    return tf.random.uniform((1, 1, 2, 2, 3), dtype=tf.float32)

