# tf.zeros(shape=(1, 20), dtype=tf.int64) ← Input shape inferred from standalone code example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model returns a constant tensor of zeros (shape 250x1, dtype int32), independent of input.
        # This mimics the SimpleModel.infer_tflite behavior in the issue.
        # Note: Such a model with no operators leads to invalid TFLite graphs as discussed.

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None], dtype=tf.int64)])
    def call(self, features):
        # Returns a constant zero tensor regardless of input
        return tf.zeros((250, 1), dtype=tf.int32)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching shape [1, 20] and dtype tf.int64, as used in the example
    return tf.random.uniform((1, 20), minval=0, maxval=10, dtype=tf.int64)

