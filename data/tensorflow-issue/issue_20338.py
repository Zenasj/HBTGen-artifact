# tf.random.uniform((B, 3, 3), dtype=tf.float32)  <- inferred input shape from example (batch size variable)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # SubModel equivalent: a custom submodel with a Dense layer of 3 units
        self.submodel = SubModel()

    def call(self, inputs):
        # Simply delegates to submodel
        return self.submodel(inputs)

class SubModel(tf.keras.Model):
    def __init__(self):
        super(SubModel, self).__init__()
        self.layer = tf.keras.layers.Dense(3)

    def call(self, inputs):
        return self.layer(inputs)

    def compute_output_shape(self, input_shape):
        # Wrap output shape using tf.TensorShape as workaround for Keras functional output shape inference bugs
        # Output shape: (batch_size, 3), assuming last dimension is 3 as Dense layer units=3
        # input_shape example: (batch_size, 3, 3)
        # Dense applies on last dim so input (3,3) -> output (3), last dim replaced by units
        return tf.TensorShape((input_shape[0], 3,))

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tensor input that matches expected input shape:
    # based on examples, input shape is (batch_size, 3, 3) of floats
    # Use batch size 5 as example
    batch_size = 5
    input_shape = (batch_size, 3, 3)
    # Uniform random float32 tensor to match MyModel expected input
    return tf.random.uniform(input_shape, dtype=tf.float32)

