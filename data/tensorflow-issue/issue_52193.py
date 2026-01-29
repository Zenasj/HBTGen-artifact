# tf.random.uniform((2, 2), dtype=tf.float32) ← input shape inferred from examples like tf.zeros([2, 2])

import tensorflow as tf

# This rewritten code fuses the concepts from the issue:
# - Handling of a Python non-Tensor argument (b) alongside Tensor input (a)
# - Usage of tf.function with tf.custom_gradient
# - The workaround demonstrated by closing over the non-Tensor argument in the custom gradient


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create an internal list of TensorWrapper objects wrapping fixed tensors
        # This mimics the issue example where non-Tensor Python objects reference underlying tensors
        self.TW_list_tf = [self.TensorWrapper(tf.zeros([2, 2])), self.TensorWrapper(tf.ones([2, 2]))]

    class TensorWrapper:
        def __init__(self, tensor):
            self._tensor = tensor  # Encapsulated tensor, no direct access assumed externally

        def plus(self, other):
            # Only public API: add other to internal tensor
            return other + self._tensor

        def __str__(self):
            return "TensorWrapper(" + str(self._tensor) + ")"

        __repr__ = __str__

    def u_tf(self, a, b):
        # Use the TensorWrapper list to add tensor 'a' to the tensor wrapped by TW_list_tf[b]
        # Note: b is a python int indexing the list, not a tensor
        return tf.reduce_sum(self.TW_list_tf[b].plus(a))

    @tf.function
    def u_tf_jit(self, a, b):
        # Jitted version of u_tf
        return self.u_tf(a, b)

    def u_tf_grad_v2(self, a, b):
        # custom_gradient with non-Tensor argument 'b' closed over in outer tf.function
        # This follows the closure fix described in the issue comments

        @tf.custom_gradient
        def inner(a):
            r = self.u_tf_jit(a, b)

            def grad(dr):
                return 2.0 * dr * tf.ones_like(a)  # no gradient wrt b

            return r, grad

        return inner(a)

    @tf.function
    def call_with_custom_gradient(self, a, b):
        # Function that uses the custom gradient function, keeping b as a python int and not Tensor
        return self.u_tf_grad_v2(a, b)

    def call(self, inputs):
        # Assume inputs is a tuple (a, b)
        a, b = inputs
        # Forward pass calls the custom gradient function with inputs
        return self.call_with_custom_gradient(a, b)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a tuple (a, b)
    # a: random float32 tensor shape (2, 2)
    # b: int index 0 or 1 to select which TensorWrapper to use, must be python int (not tensor)
    a = tf.random.uniform((2, 2), dtype=tf.float32)
    b = 1  # fixed to 1 as per example usage; could be randomized via Python random choice but fixed here
    return (a, b)

