# tf.random.uniform((B,)) ← Input is a 1D tensor with unknown shape (batch size undefined), inferred from tf.keras.Input with shape=0 (scalar) in example

import tensorflow as tf

class SqueezedSparseConversion(tf.keras.layers.Layer):
    def call(self, inputs):
        # Constructs a fixed sparse tensor ignoring inputs, shape (3,3)
        return tf.SparseTensor(indices=[[0, 1]],
                               values=[0.1],
                               dense_shape=[3, 3])

class GraphConvolution(tf.keras.layers.Layer):
    def call(self, inputs):
        # Inputs is expected to be a list/tuple: [dense tensor, sparse tensor]
        # This example just returns the dense tensor (inputs[0]) as output
        return inputs[0]

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Encapsulate the two layers from the example
        self.sparse_layer = SqueezedSparseConversion()
        self.graph_conv = GraphConvolution()

    def call(self, inputs):
        # inputs: dense tensor of arbitrary batch size and no shape specified (scalar per batch element)
        # 1. Create sparse tensor from input using SqueezedSparseConversion
        sp = self.sparse_layer(inputs)
        # 2. Pass both dense input and sparse tensor to GraphConvolution
        out = self.graph_conv([inputs, sp])
        return out

def my_model_function():
    # Instantiate MyModel
    model = MyModel()
    # Build the model by calling it once with sample input of shape (1,)
    sample_input = GetInput()
    _ = model(sample_input)
    return model

def GetInput():
    # Return a random input tensor that matches the expected input shape:
    # Scalar per batch element, unknown batch size: shape (1,)
    # This aligns with tf.keras.Input(0) in the original example, which is scalar input with unknown batch size.
    # We'll produce a batch dimension of 1 here.
    return tf.random.uniform((1,), dtype=tf.float32)

