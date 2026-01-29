# tf.random.uniform((2, 17, 17, 768), dtype=tf.float32) ← inferred input shape from the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # According to the issue, MaxPooling2D is constructed with a very large pool_size [1e+20, True]
        # which is invalid and leads to crashes.
        # We'll interpret intention as a large pooling window on a 2D input.
        # True is invalid for pool_size, so we replace with a typical valid large integer pooling size.
        # We assume pool_size should be an int or tuple of 2 ints.
        # Use a large but reasonable pool size to replicate intended stress:
        
        # Interpreting the problematic input as pool_size=[1e20, True]
        # We replace with pool_size=(17,17) (full spatial extent), typical for input 17x17.
        # strides are [2,2] as per the issue.
        self.pool_size = (17, 17)
        self.strides = (2, 2)
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=self.pool_size,
                                                     strides=self.strides)
        
    def call(self, inputs):
        # Single input tensor of shape (2, 17, 17, 768)
        return self.max_pool(inputs)

def my_model_function():
    # Return an instance of MyModel.
    # No special weights to load, so simple initialization.
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected input shape for MyModel:
    # batch size=2, height=17, width=17, channels=768, dtype=float32.
    return tf.random.uniform(shape=(2, 17, 17, 768), dtype=tf.float32)

