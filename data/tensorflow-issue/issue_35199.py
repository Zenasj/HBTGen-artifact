# tf.random.uniform((1, 256, 256, 3), dtype=tf.float32) ← inferred input shape based on the example Input layer shape=[256, 256, 3]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define an Input layer for clarity (though not strictly necessary in a Subclassed model)
        # We will implement softmax in two ways and compare them:
        # - Using keras.activations.softmax which includes the original multi-dim workaround
        # - Using tf.nn.softmax directly (the preferred method for TFLite conversion)
        
    def call(self, inputs):
        # Apply keras.activations.softmax (which has multi-dim workaround)
        # This is effectively equivalent to:
        #   e = exp(x - reduce_max(x, axis))
        #   s = reduce_sum(e, axis)
        #   e / s
        softmax_keras = tf.keras.activations.softmax(inputs)
        
        # Apply tf.nn.softmax directly - this is the native TF op with multi-dim support
        softmax_tf_nn = tf.nn.softmax(inputs)
        
        # Compare the outputs numerically (close within some tolerance)
        # This highlights the difference in implementations causing TFLite conversion issues
        # The output is a boolean tensor where they are close
        comparison = tf.math.abs(softmax_keras - softmax_tf_nn) < 1e-6
        
        # For demonstration, return all three:
        # (softmax using keras.activations.softmax, softmax using tf.nn.softmax, comparison mask)
        return softmax_keras, softmax_tf_nn, comparison

def my_model_function():
    # Return an instance of MyModel.
    return MyModel()

def GetInput():
    # Generate a random float32 tensor with shape (1, 256, 256, 3) matching the Input layer in the original example
    # batch size = 1, height=256, width=256, channels=3
    return tf.random.uniform((1, 256, 256, 3), dtype=tf.float32)

