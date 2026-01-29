# tf.random.uniform((2, 10), dtype=tf.uint16)  ← representative shape and dtype example for the largest tensor in the set

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Since the original issue revolves around nested data structures with multiple tensors
        # of different shapes and dtypes wrapped in ExtensionTypes, we'll simulate that with
        # explicit tf.keras layers holding these tensors as variables or constants.
        # The forward pass will just return a dict of these tensors for demonstration.
        # This is a "fused" model encapsulating multiple tensor specs and types.

        # Following the patterns in the issue, we maintain shapes and dtypes exactly.
        # These can be considered as "loaded" tensors or model state.

        # Using tf.Variables to mimic fixed shapes and dtypes in model subcomponents.
        self.b1 = tf.Variable(tf.ones(shape=[2, 1], dtype=tf.float32), trainable=False, name="b1")     # (2,1), float32
        self.b2 = tf.Variable(tf.ones(shape=[2, 2], dtype=tf.float16), trainable=False, name="b2")    # (2,2), float16
        self.b3 = tf.Variable(tf.ones(shape=[2, 3], dtype=tf.int8), trainable=False, name="b3")       # (2,3), int8
        self.b4 = tf.Variable(tf.ones(shape=[2, 4], dtype=tf.int16), trainable=False, name="b4")      # (2,4), int16
        self.b5 = tf.Variable(tf.ones(shape=[2, 5], dtype=tf.int32), trainable=False, name="b5")      # (2,5), int32
        self.b6 = tf.Variable(tf.ones(shape=[2, 6], dtype=tf.bfloat16), trainable=False, name="b6")   # (2,6), bfloat16
        self.b7 = tf.Variable(tf.ones(shape=[2, 7], dtype=tf.float64), trainable=False, name="b7")    # (2,7), float64
        self.b8 = tf.Variable(tf.ones(shape=[2, 8], dtype=tf.int64), trainable=False, name="b8")      # (2,8), int64
        self.b9 = tf.Variable(tf.ones(shape=[2, 9], dtype=tf.uint8), trainable=False, name="b9")      # (2,9), uint8
        self.b10 = tf.Variable(tf.ones(shape=[2, 10], dtype=tf.uint16), trainable=False, name="b10")  # (2,10), uint16

        # Nested sub-structure - A with a1 tensor
        self.a1 = tf.Variable(tf.ones(shape=[2, 11], dtype=tf.float16), trainable=False, name="a1")   # (2,11), float16

    def call(self, inputs):
        # The input is not used for computation since original code loads fixed tensors.
        # Return a dict similar to the ExtensionType B holding its tensors including nested A.

        # For possible comparison or demonstration,
        # output all tensors as a dict keyed by their names.
        return {
            "b1": self.b1,
            "b2": self.b2,
            "b3": self.b3,
            "b4": self.b4,
            "b5": self.b5,
            "b6": self.b6,
            "b7": self.b7,
            "b8": self.b8,
            "b9": self.b9,
            "b10": self.b10,
            "a": {"a1": self.a1}
        }


def my_model_function():
    # Return an instance of MyModel as required.
    return MyModel()


def GetInput():
    # According to the original example, MyModel does not require input to produce outputs,
    # but we must provide compatible input to call the model.
    # Since model.forward doesn't use input, return a dummy tensor of shape that could mimic batch input. 
    # We'll return a random tensor of shape (1, 1) float32, as the inputs are unused.
    return tf.random.uniform((1, 1), dtype=tf.float32)

