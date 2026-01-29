# tf.range(600), tf.random.uniform((600,), dtype=tf.float64) ← Input shape inferred from original model producing 600-length vectors

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters; logic directly creates tensors

    @tf.function
    def call(self, inputs=None):
        # Inputs is unused, as original callable produces fixed-length outputs
        
        # frames: 1D int32 tensor [0..599]
        frames = tf.range(600, dtype=tf.int32)
        
        # bpm: random uniform float64 tensor shape (600,) value range [0, 90)
        bpm = tf.random.uniform(
            shape=(600,),
            minval=0,
            maxval=90,
            dtype=tf.float64
        )
        
        # Return the two tensors as a tuple
        return bpm, frames

def my_model_function():
    # Return an instance of MyModel (no special initialization needed)
    return MyModel()

def GetInput():
    # Since the original model has no inputs, create a dummy input tensor to satisfy TF Keras signature
    # Use a dummy tensor of shape (1,) int32 zero as a placeholder input
    return tf.zeros((1,), dtype=tf.int32)

