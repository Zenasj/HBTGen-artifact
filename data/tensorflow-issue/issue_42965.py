# tf.random.uniform((B, ), dtype=tf.float32) where B=2, matching input1_shape=(2,) and input2_shape=(1,)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using Multiply layer as in the described model, supports broadcasting
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        # inputs is expected to be a list/tuple of two tensors: [input1, input2]
        input1, input2 = inputs
        # Perform elementwise multiply with broadcasting support
        return self.multiply([input1, input2])

def my_model_function():
    # Return an instance of MyModel. No special weights initialization is needed.
    return MyModel()

def GetInput():
    # Based on the issue:
    # input1 shape = (2,)
    # input2 shape = (1,) for broadcast testing case
    # Use float32 inputs (as this is a Keras model, not quantized input)
    # Values randomly chosen in range [0,3] to reflect quantized_input_stats scale in example
    input1 = tf.random.uniform(shape=(2,), minval=0.0, maxval=3.0, dtype=tf.float32)
    input2 = tf.random.uniform(shape=(1,), minval=0.0, maxval=3.0, dtype=tf.float32)
    return [input1, input2]

