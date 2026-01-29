# tf.random.uniform((1, 10), dtype=tf.float32) ← Input shape inferred from the original examples concatenating along axis=1 with shape (1,10)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A constant tensor similar to the issue case, shape (1, 10)
        # Using a tf.constant to simulate the constant tensor concatenated in the original issue.
        # This matches the shape and datatype from the example in the issue.
        self.const_tensor = tf.constant(tf.zeros((1, 10), dtype=tf.float32))
    
    @tf.function
    def call(self, input_tensor):
        # The original issue describes concatenation of two constant / variable tensors with the input.
        # Since the GPU TFLite delegate fails to concat if any tensor is constant with shape not BHWC,
        # the model here replicates the concatenation of input with a constant along axis=1 (feature axis in 2D tensor).
        #
        # Input tensor shape: (1, 10)
        #
        # Concatenate input tensor and constant tensor along axis=1 (second dimension).
        # Output shape: (1, 20)
        return tf.concat([input_tensor, self.const_tensor], axis=1)

def my_model_function():
    # Return an instance of MyModel.
    return MyModel()

def GetInput():
    # Generate a valid random input tensor shaped (1, 10) matching the input expected by MyModel.
    # Using uniform distribution here, dtype float32.
    return tf.random.uniform((1, 10), dtype=tf.float32)

