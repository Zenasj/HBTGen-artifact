# tf.random.uniform((1, 1, 3), dtype=tf.uint8) ← inferred from input_signature shape and dtype in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No layers as per the original example, model just returns input unchanged
        # This mimics the Net class in the issue where process returns image as is
        # This minimal model structure is required for saved_model export and TFLite conversion
        
    @tf.function(input_signature=[tf.TensorSpec(shape=(1,1,3), dtype=tf.uint8)])
    def call(self, image):
        return image


def my_model_function():
    # Return an instance of MyModel
    # No special initialization or weights since original Net was identity function
    return MyModel()

def GetInput():
    # Return a random tensor matching the expected input shape and dtype
    return tf.random.uniform(shape=(1, 1, 3), minval=0, maxval=255, dtype=tf.uint8)

