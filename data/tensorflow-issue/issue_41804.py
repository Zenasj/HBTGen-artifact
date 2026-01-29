# tf.random.uniform((1, 512, 2), dtype=tf.float32), tf.random.uniform((1, 1), maxval=512, dtype=tf.int64)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters needed; tf.gather is a TF built-in op.
        pass

    def call(self, inputs):
        # inputs: a tuple (data, indices)
        # data shape: [1, 512, 2]
        # indices shape: [1, 1] (int64)
        data, indices = inputs
        # Perform tf.gather with batch_dims=1, replicating the example.
        # According to the original issue:
        # TensorFlow (graph) output shape: (1, 1, 2)
        # TFLite output shape error: (1, 1, 1, 2)
        output = tf.gather(data, indices, batch_dims=1)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a tuple of inputs matching the expected input signatures:
    # data: float32 tensor of shape [1, 512, 2]
    # indices: int64 tensor of shape [1, 1], values in [0,512)
    data = tf.random.uniform((1, 512, 2), dtype=tf.float32)
    indices = tf.random.uniform((1, 1), minval=0, maxval=512, dtype=tf.int64)
    return (data, indices)

