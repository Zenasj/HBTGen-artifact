# tf.random.uniform((BATCH_SIZE, INPUT_SIZE), dtype=tf.float32)
import tensorflow as tf
import numpy as np

# Assumptions from the issue:
# - Input shape is (batch_size, 3) since INPUT_SIZE=3 in the example
# - Output from Dense layer is size 2
# - Model uses a single Dense layer with weights set to all ones and bias zeros
# - We replicate the simple model and weights initialization described
# - For the input generator, we produce a batch of samples with shape (batch, 3)
# - The class will include the Dense layer and setting weights as in the example
# - We do not implement Sequence or multi-processing logic (that is external to the model)
# - We include comments about this being compatible with TF 2.20.0 and jit_compile

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(2)
        # Initialize weights to match the issue demo: weights all ones, bias zero
        # We defer actual assignment to build since layer is not built yet

    def build(self, input_shape):
        super().build(input_shape)
        # weights shape: (INPUT_SIZE=3, DENSE_OUTPUTS=2)
        w_shape = (input_shape[-1], 2)
        b_shape = (2,)
        # Initialize weight matrix to all ones, bias to zeros
        weights = tf.ones(w_shape, dtype=self.dtype)
        bias = tf.zeros(b_shape, dtype=self.dtype)
        self.dense.set_weights([weights.numpy(), bias.numpy()])  # need numpy array or EagerTensor

    def call(self, inputs):
        return self.dense(inputs)


def my_model_function():
    # Return an instance of MyModel with initialized weights
    model = MyModel()
    # Build the model explicitly by passing a dummy input
    dummy_input = tf.zeros((1, 3), dtype=tf.float32)
    model(dummy_input)  # triggers build and sets weights
    return model


def GetInput():
    # Return random input tensor matching batch size x input dimension
    # We pick batch size 2 to match issue example batch_size=2
    batch_size = 2
    input_size = 3
    # dtype float32 as per default tf.keras layers
    return tf.random.uniform((batch_size, input_size), dtype=tf.float32)

