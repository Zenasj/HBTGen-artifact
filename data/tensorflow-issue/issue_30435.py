# tf.random.uniform((B, H, W, C), dtype=...) ← Input shape is ambiguous from the issue, but the example works with a scalar int input iterating over a list of length 3

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Instead of python list, use a tf.constant for indexing within tf.while_loop
        self.lst = tf.constant([1, 2, 3], dtype=tf.int32)
        self.total_length = tf.size(self.lst)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs expected: a scalar tf.int32 tensor, initial counter (usually 0)
        i = inputs

        # This list will accumulate printed values during the loop for demonstration
        output = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        def cond(i, out):
            return tf.less(i, self.total_length)

        def body(i, out):
            # Print current counter (similar to tf.print in TF2)
            tf.print("Loop counter i:", i)
            current_value = self.lst[i]
            tf.print("Current list value:", current_value)

            # Write current_value to output array to observe the values iterated
            out = out.write(i, current_value)
            i = i + 1
            return i, out

        i, output = tf.while_loop(cond, body, loop_vars=[i, output])
        # Stack all outputs collected
        result = output.stack()
        return result

def my_model_function():
    return MyModel()

def GetInput():
    # Return initial counter scalar tensor 0 for the loop starting index
    return tf.constant(0, dtype=tf.int32)

