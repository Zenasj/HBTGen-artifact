# tf.random.uniform((1, 10), dtype=tf.float32) ← inferred input shape from example usage min_value/max_value of length 10

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No specific layers needed since this model replicates the "foo_with_custom_attributes" logic
        # to preserve attributes when using tf.function

    @tf.function
    def call(self, x_label_tuple):
        # Unpack input tuple
        label, tensor, min_value, max_value = x_label_tuple

        # We can't directly set attributes inside tf.function due to tracing limitations,
        # but we can produce output tensor, then outside of tf.function attach the attribute.
        # So we embed the call to an inner tf.function that returns the tensor,
        # then attach the attribute outside call().

        @tf.function
        def inner_func(label, tensor, min_value, max_value):
            if tensor is not None and tf.is_tensor(tensor):
                result = tf.identity(tensor, name=label)
            else:
                # Use uniform random with given min and max values shapes
                result = tf.random.uniform(shape=(1, tf.shape(min_value)[0]),
                                           minval=min_value, maxval=max_value,
                                           name=label)
            return result

        result = inner_func(label, tensor, min_value, max_value)

        # Attach custom attribute outside tf.function tracing
        # Note: attributes will not be preserved inside compiled graph execution, but this simulates
        # the suggested workaround pattern to attach metadata externally.
        # We store it as a private attribute "_bar" similarly to the example.
        result._bar = [label]

        return result


def my_model_function():
    """
    Returns an instance of MyModel.
    This model expects a tuple of (label:str tensor,
                                   input tensor or None,
                                   min_value tensor,
                                   max_value tensor)
    """
    return MyModel()


def GetInput():
    """
    Creates sample input data tuple matching MyModel call signature:

    label: string tensor
    tensor: None (simulates no tensor input)
    min_value: tensor of shape (10,), float32
    max_value: tensor of shape (10,), float32
    """
    # Using tf.constant for label as a scalar string tensor
    label = tf.constant("a")

    # tensor=None to trigger random uniform generation
    tensor = None

    min_value = tf.zeros((10,), dtype=tf.float32)  # lowest 10 dims values are 0
    max_value = tf.ones((10,), dtype=tf.float32)   # highest 10 dims values are 1

    return (label, tensor, min_value, max_value)

