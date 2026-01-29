# tf.ragged.constant([[...], [...]], dtype=int32) → Input is a RaggedTensor with shape (batch_size=2, variable_inner_dims)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # For demonstration, we'll implement a simple model that accepts
        # a RaggedTensor input, converts it to a dense tensor by padding,
        # and then applies a simple Dense layer.

        # Note: Since tf.data.Dataset.from_generator has issues with RaggedTensor outputs,
        # the model illustrates safe handling by converting ragged input into dense
        # with padding for downstream layers.

        # We'll accept ragged input and produce some output.
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs):
        # inputs are expected as a RaggedTensor of shape (batch, None, None ...)
        # For simplicity, convert ragged inputs to dense with padding (0)
        # This is a common approach when working with ragged data inside models.
        dense_inputs = inputs.to_tensor(default_value=0)

        # Apply a dense layer on last dimension if applicable.
        # If input is 2D or higher, Dense applies on last axis.
        output = self.dense(dense_inputs)
        return output

def my_model_function():
    # Return an instance of MyModel.
    return MyModel()

def GetInput():
    # Returns a RaggedTensor input compatible with MyModel.
    # As illustrated in the issue, ragged tensors with
    # nested variable lengths are common.

    # Example batch size 2:
    # First example has 2 inner lists: [1,2] and [1]
    # Second example has 1 inner list: [3,4,5]

    ragged_input = tf.ragged.constant([
        [[1, 2], [1]],
        [[3, 4, 5]]
    ], dtype=tf.int32)

    return ragged_input

