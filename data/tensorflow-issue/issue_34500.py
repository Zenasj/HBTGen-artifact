# tf.constant([], dtype=tf.int32) ← Input is a scalar integer n representing the range limit

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No additional layers needed since this is computational logic.

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)], experimental_compile=True)
    def call(self, n):
        # Implements the slow control flow FizzBuzz counting logic from the issue
        fizz = tf.constant(0, dtype=tf.int32)
        buzz = tf.constant(0, dtype=tf.int32)
        fizzbuzz = tf.constant(0, dtype=tf.int32)
        
        # Because Python range is not graph compatible, use tf.while_loop
        i = tf.constant(0, dtype=tf.int32)

        def cond(i, fizz, buzz, fizzbuzz):
            return i < n

        def body(i, fizz, buzz, fizzbuzz):
            fizzbuzz_inc = tf.cond(tf.equal(tf.math.mod(i, 6), 0),
                                   lambda: fizzbuzz + 1,
                                   lambda: fizzbuzz)
            buzz_inc = tf.cond(
                tf.logical_and(tf.not_equal(tf.math.mod(i, 6), 0),
                               tf.equal(tf.math.mod(i, 3), 0)),
                lambda: buzz + 1,
                lambda: buzz)
            fizz_inc = tf.cond(
                tf.logical_and(
                    tf.logical_and(tf.not_equal(tf.math.mod(i, 6), 0),
                                   tf.not_equal(tf.math.mod(i, 3), 0)),
                    tf.equal(tf.math.mod(i, 2), 0)),
                lambda: fizz + 1,
                lambda: fizz)
            return i + 1, fizz_inc, buzz_inc, fizzbuzz_inc

        _, fizz_out, buzz_out, fizzbuzz_out = tf.while_loop(cond, body, [i, fizz, buzz, fizzbuzz])
        # Return as a tensor of shape [3], aligning with original spec
        return tf.stack([fizz_out, buzz_out, fizzbuzz_out])

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Return a scalar tensor input representing n, e.g. 100000 for the benchmark
    # Chosen 100000 as per issue benchmark inputs
    return tf.constant(100000, dtype=tf.int32)

