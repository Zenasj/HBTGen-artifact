# tf.Tensor(shape=(), dtype=int64) ← The input is expected to be a tf.data.Dataset of scalar int64 tensors

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model will wrap a tf.data.Dataset input and iterate over it,
        # outputting a list (tensor array) of elements iterated.
        # We will do this to replicate the logic of iterating over dataset inside tf.function.
        # Since Dataset iteration returns tensors scalar or shape=(), we'll accumulate them 
        # into a tf.Tensor array to produce a tensor output from call().
        # This is a basic iterable collector model, demonstrating safe dataset iteration inside tf.function.
    
    @tf.function
    def call(self, dataset):
        # Accumulate dataset elements in a TensorArray to output all at once
        output = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        idx = tf.constant(0)

        # Use tf.data.experimental.get_single_element when dataset is single element,
        # but here we want iteration to collect all elements.
        # We can't use python iteration in tf.function easily for datasets with unknown length,
        # so we use tf.data.Iterator in graph mode by creating an iterator and looping.
        # The safest way: use tf.data.Iterator get_next via tf.while_loop.

        # Create initial iterator
        iterator = iter(dataset)

        def cond(idx, output):
            # Continue indefinitely, will catch StopIteration to break
            # Use try/except in tf.function by tf.py_function workaround
            return True

        def body(idx, output):
            # Try to get next dataset element
            # We must handle StopIteration, which is raised when dataset runs out
            # Use tf.py_function to safely get next element and flag when exhausted
            def get_next_py():
                try:
                    element = next(iterator)
                    return element, False
                except StopIteration:
                    return tf.constant(-1, dtype=tf.int64), True  # dummy value and stop flag

            element, stop = tf.py_function(get_next_py, [], [tf.int64, tf.bool])
            element.set_shape(())
            stop.set_shape(())
            # Condition to stop iteration
            if tf.cast(stop, tf.bool):
                # Break
                return idx, output
            # Else, write element to output
            output = output.write(idx, element)
            idx += 1
            return idx, output

        # We iterate up to an arbitrary max to avoid infinite loop if something goes wrong
        max_iter = 1000

        def cond_loop(idx, output):
            return tf.less(idx, max_iter)

        def loop_body(idx, output):
            idx_new, output_new = body(idx, output)
            # Stop condition inside body leads to no increment, so stop by breaking loop with idx_new == idx
            return tf.cond(
                tf.equal(idx_new, idx),
                lambda: (max_iter, output_new),  # forcibly terminate loop
                lambda: (idx_new, output_new),
            )

        idx_final, output_final = tf.while_loop(cond_loop, loop_body, [idx, output])

        return output_final.stack()

def my_model_function():
    # Return an instance of MyModel (no weights or custom init needed)
    return MyModel()

def GetInput():
    # Return a tf.data.Dataset of int64 scalars, compatible with MyModel's call input
    # Using tf.data.Dataset.range(5) like in the original example
    return tf.data.Dataset.range(5)

