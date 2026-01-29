# tf.random.uniform((2, 4, 8, 2), dtype=tf.uint16) ← Input is a 4D tensor with shape (batch=2, height=4, width=8, channels=2), dtype uint16

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # UpSampling2D layer with bilinear interpolation and size (2,3)
        # Note: Input dtype is uint16, but UpSampling2D historically converted unsigned int input 
        # to float output (e.g., float32). In TF 2.17, output dtype changed to same as input.
        # To reproduce the old behavior, we add an explicit cast to float32 before upsampling.
        self.cast_to_float = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 3), data_format='channels_last', interpolation='bilinear')
        
    def call(self, inputs):
        # To be consistent with prior behavior (output float32 for uint inputs), 
        # cast input to float32 before upsampling
        x = self.cast_to_float(inputs)
        y = self.upsample(x)
        return y

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random uint16 tensor matching input shape [2, 4, 8, 2]
    # Values range [-8, 8) clipped to fit uint16, which cannot have negative values.
    # For realistic uint16 input, use 0 to 15 range instead of -8 to 8.
    # We keep the same shape and datatype as used in the issue description.
    # Since negative values are not representable in uint16, we shift range positive.
    import numpy as np
    rng = np.random.default_rng()
    # Generate integers in [0, 16) to mimic 0-centered range shifted from [-8,8)
    data_np = rng.integers(0, 16, size=(2, 4, 8, 2), dtype=np.uint16)
    return tf.convert_to_tensor(data_np, dtype=tf.uint16)

