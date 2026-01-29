# tf.random.uniform((1,), dtype=tf.int32) ← Inferred input shape: scalar integer with batch_size=1

import tensorflow as tf
from tensorflow.python.framework import func_graph
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The Lambda layer wraps lambda_fn with a special output shape inference workaround
        # to allow building the model in eager mode without OperatorNotAllowedInGraphError.
        self.lambda_layer = tf.keras.layers.Lambda(
            self.lambda_fn, 
            output_shape=self.infer_output_shape(self.lambda_fn),
            dynamic=True  # dynamic=True here indicates layer runs eagerly.
        )

    def call(self, inputs, training=None):
        return self.lambda_layer(inputs)

    def lambda_fn(self, inputs):
        # This function processes inputs with Python control flow (a while loop)
        # that iterates from 0 to inputs-1, printing each iteration.
        # It returns inputs as output.

        i = tf.constant(0, dtype=tf.int32)
        cond = lambda i: i < inputs
        
        # Use tf.while_loop with autograph to ensure graph-compatibility and eager debugging capability.
        def body(i):
            tf.print("loop iteration", i)
            return i + 1
        
        i = tf.while_loop(cond, body, [i])

        return inputs

    def infer_output_shape(self, fn):
        # Use autograph conversion on an ephemeral graph to infer output shape of lambda_fn.
        # This helps the Lambda layer avoid run-time shape inference that triggers errors in eager mode.
        # Note: input is a placeholder tensor simulating actual model input shape.

        @tf.function
        def wrapped(input_tensor):
            return fn(input_tensor)

        scratch_graph = func_graph.FuncGraph("scratch_graph")
        with scratch_graph.as_default():
            # Create a placeholder tensor with the expected input shape and dtype
            # KerasTensor has a _to_placeholder method that creates a TF placeholder consistent with inputs.
            # Since we don’t have the real input yet at this point, create a dummy input here.
            dummy_input = tf.keras.Input(batch_size=1, shape=(), dtype=tf.int32)
            output = fn(dummy_input)
        return output.shape


def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random tensor with shape (1,) and dtype int32,
    # consistent with batch_size=1, scalar input expected by MyModel.
    # Use tf.random.uniform with minval=0 and maxval=5 (arbitrary positive range) to avoid negative or zero input.
    return tf.random.uniform(shape=(1,), minval=0, maxval=5, dtype=tf.int32)

