# tf.random.uniform((1,), dtype=tf.int32) ← inferred input shape for element_shape tensor

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters, this model wraps the list_ops.empty_tensor_list call
        # which expects element_shape as a 1-D int32 tensor specifying the shape of elements.
        # Based on the issue, element_shape MUST have rank 1, not rank 2.
    
    @tf.function(jit_compile=True)
    def call(self, element_shape):
        # This models the behavior of list_ops.empty_tensor_list with validation on element_shape
        # We emulate the check that element_shape must be a 1-D tensor,
        # and raise a ValueError if rank is more than 1, reflecting the fix for the bug.
        
        # Note: The real tensorflow implementation crashes in C++ if rank != 1,
        # here we intercept it and throw a Python error for clarity.
        
        # In actual TF code, element_shape would be a tf.Tensor of shape [rank].
        shape_rank = tf.rank(element_shape)
        # If rank != 1, raise an error.
        def true_fn():
            msg = tf.strings.format(
                "Shape must be at most rank 1 but is rank {} [Op:EmptyTensorList]", shape_rank)
            # Raise in graph mode by tf.debugging.assert
            tf.debugging.assert_equal(False, True, message=msg)
            return tf.constant([], dtype=tf.int32)  # Dummy (won't reach here)
        
        def false_fn():
            # Call the actual TF operation to create empty tensor list
            return tf.raw_ops.EmptyTensorList(element_shape=element_shape, element_dtype=tf.int32)
        
        return tf.cond(shape_rank > 1, true_fn, false_fn)


def my_model_function():
    # Return an instance of MyModel for use.
    return MyModel()


def GetInput():
    # Return a 1-D int32 tensor input for element_shape that is valid.
    # Because the bug relates to element_shape having rank 2 (e.g. [[1]]) causing crash,
    # we provide rank 1 shaped tensor like [1].
    #
    # Here we generate a random 1-D shape tensor of length 1 (i.e. shape=[1]),
    # with a single integer dimension size between 1 and 10.
    
    element_shape = tf.random.uniform(shape=(1,), minval=1, maxval=10, dtype=tf.int32)
    return element_shape

