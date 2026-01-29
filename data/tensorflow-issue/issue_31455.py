# tf.random.uniform((B, 1), dtype=tf.float32) for each of two inputs

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two inputs, each (None, 1) float32, that get concatenated to (None, 2)
        self.concat = tf.keras.layers.Lambda(lambda inputs: tf.concat(inputs, axis=-1))
        self.dense = tf.keras.layers.Dense(3)
        # Instead of creating Lambda layers inside a loop (which captures late-binding variables),
        # we create fixed lambda layers for each output slice, to avoid the bug described in the issue.
        self.output_slices = [
            tf.keras.layers.Lambda(lambda outputs, i=i: outputs[..., i], name=f'output_{i}')
            for i in range(3)
        ]

    def call(self, inputs, training=False):
        # inputs: list or tuple of two tensors, each (B,1)
        x = self.concat(inputs)
        x = self.dense(x)
        # Return list of outputs, each slicing a different last-dim channel of dense output
        outputs = [slice_layer(x) for slice_layer in self.output_slices]
        return outputs

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a list of two tensors, each with shape (batch_size=2, 1)
    # Matching the shape expected by MyModel
    # Matching the example input in the issue, but can be random floats instead
    batch_size = 2
    input0 = tf.random.uniform((batch_size, 1), minval=-1, maxval=1, dtype=tf.float32)
    input1 = tf.random.uniform((batch_size, 1), minval=-1, maxval=1, dtype=tf.float32)
    return [input0, input1]

