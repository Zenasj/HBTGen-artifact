# tf.random.uniform((18, 54, 6, 6), dtype=tf.float32) ← inferred as main input shape based on second example inputs

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Based on both examples from the issue,
        # we fuse the two conv2d + tan models as submodules in MyModel.
        # The first example uses inputs p0 and p1 (stored as self.p0 and self.p1),
        # shapes: p0: [6, 21, 59, 6], p1: [1, 54, 6, 6]
        # The second example takes a single input and conv2ds it with stored p0.
        # We'll embed both convolution kernels as variables/constants in this fused model.

        # From second example (Chunk 2), stored p0 and p1:
        self.p0 = tf.random.uniform(shape=[6, 21, 59, 6], dtype=tf.float32)
        self.p1 = tf.random.uniform(shape=[1, 54, 6, 6], dtype=tf.float32)

        # From first example (Chunk 1):
        # The inputs are two tensors of shape:
        # inp1: [1, 8147, 1, 1] and inp2: [1, 1, 8160, 1]
        # We'll include this conv call as a method but keep it separate
        # from the example 2 conv.

    @tf.function(jit_compile=True)
    def __call__(self, inputs):
        # inputs: a tuple or list of tensors
        # Based on usage, to accommodate both examples, 
        # we accept either:
        # - inputs = (inp1, inp2) for first example conv2d: tf.nn.conv2d(inp2, inp1, ...)
        # - or inputs = (inp,) for second example conv2d: tf.nn.conv2d(self.p0, inp, ...)

        # To unify these two patterns, we branch on input length

        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            # First example mode
            inp1, inp2 = inputs
            # conv2d with stride=1, padding=VALID, dilation=(2,1)
            conv1 = tf.nn.conv2d(inp2, inp1, strides=1, padding="VALID", dilations=(2,1))
            tan1 = tf.tan(conv1)
            return conv1, tan1

        elif isinstance(inputs, (list, tuple)) and len(inputs) == 1:
            # Second example mode
            inp = inputs[0]
            # conv2d with stride=1, padding=SAME, dilation=(3,3)
            conv2 = tf.nn.conv2d(self.p0, inp, strides=1, padding="SAME", dilations=(3,3))
            tan2 = tf.tan(conv2)
            return conv2, tan2

        else:
            # Unsupported input scheme: raise error with explanation
            raise ValueError(
                "MyModel expects inputs as either tuple of two tensors (inp1, inp2) "
                "or single-element tuple (inp,). Provided input shape(s): "
                + str([tensor.shape for tensor in inputs]) if isinstance(inputs, (list, tuple)) else str(inputs)
            )

def my_model_function():
    # Return an instance of MyModel.
    # Random weights are embedded in the model itself.
    return MyModel()

def GetInput():
    # This function should return an input that can be directly passed to MyModel instance.
    # Since MyModel supports two input schemes (two tensors or one tensor),
    # we will return the input matching the second example as default,
    # because in the issue they seem to confirm it triggers the same bug.

    # From chunk 2, input shape is [18,54,6,6]
    input_tensor = tf.random.uniform(shape=[18, 54, 6, 6], dtype=tf.float32)
    # Return as single-element tuple to match second example signature
    return (input_tensor,)

