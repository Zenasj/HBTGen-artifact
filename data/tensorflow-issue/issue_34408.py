# tf.random.uniform((1, 2), dtype=tf.float32) ← inferred input shape from example in issue (input_shape=(1,2))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model encapsulates the logic of the example MyCustomLayer
        # which returns a list of tensors (duplicates input).
        # Here we implement this behavior as part of the model call.
    
    def call(self, inputs):
        # According to the issue, a layer/model that returns list/tuple of tensors is valid,
        # but keras.testing_utils.layer_test expects single tensor output.
        # Here we implement the call returning a tuple/list of the inputs as output,
        # matching the MyCustomLayer from the issue.
        return [inputs, inputs]

def my_model_function():
    # Return an instance of the model, no special initialization needed here.
    return MyModel()

def GetInput():
    # Return a random tensor of shape (1, 2) matching the input_shape in the issue,
    # dtype float32 as a reasonable default.
    return tf.random.uniform((1, 2), dtype=tf.float32)

