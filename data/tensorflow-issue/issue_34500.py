import numpy as np
import tensorflow as tf

class FizzBuzz(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def model(self,
              n  # Shape [] -- int32 the max number to loop FizzBuzz to
              ):  # Returns counts for fizz, buzz and fizzbuzz. Shape: [1] with length 3
        fizz = 0
        buzz = 0
        fizzbuzz = 0
        for i in range(n):
            if i % 6 == 0:
                fizzbuzz += 1
            elif i % 3 == 0:
                buzz += 1
            elif i % 2 == 0:
                fizz += 1
        return [fizz, buzz, fizzbuzz]

class FizzBuzz(tf.Module):
    def model(self,n):
        fizz = np.array(0)
        buzz = np.array(0)
        fizzbuzz = np.array(0)
        # Force everything to be a numpy scalar, for an even comparison
        for i in np.arange(n)[:, np.newaxis]:
            if i % 6 == 0:
                fizzbuzz += 1
            elif i % 3 == 0:
                buzz += 1
            elif i % 2 == 0:
                fizz += 1
        return [fizz, buzz, fizzbuzz]

class FizzBuzz(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)], autograph=False)
    def model(self,
              n  # Shape [] -- int32 the max number to loop FizzBuzz to
              ):  # Returns counts for fizz, buzz and fizzbuzz. Shape: [1] with length 3
        fizz = 0
        buzz = 0
        fizzbuzz = 0

        def cond(i, fizz, buzz, fizzbuzz):
          return i < n

        def body(i, fizz, buzz, fizzbuzz):
          return (i + 1,) + tf.cond(
              i % 6 == 0,
              lambda: (fizz, buzz, fizzbuzz + 1),
              lambda: tf.cond(
                  i % 3 == 0,
                  lambda: (fizz, buzz + 1, fizzbuzz),
                  lambda: tf.cond(
                      i % 2 == 0,
                      lambda: (fizz + 1, buzz, fizzbuzz),
                      lambda: (fizz, buzz, fizzbuzz)
                  )
              )
          )

        _, fizz, buzz, fizzbuzz = tf.while_loop(
            cond, body, (0, fizz, buzz, fizzbuzz))
        return [fizz, buzz, fizzbuzz]

class FizzBuzz(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def model(self,
              n  # Shape [] -- int32 the max number to loop FizzBuzz to
              ):  # Returns counts for fizz, buzz and fizzbuzz. Shape: [1] with length 3
        i = tf.range(n)
        fizz_v = (i % 2 == 0)
        buzz_v = (i % 3 == 0)
        fizz = tf.reduce_sum(tf.cast(fizz_v, tf.int32))
        buzz = tf.reduce_sum(tf.cast(buzz_v, tf.int32))
        fizzbuzz = tf.reduce_sum(tf.cast(tf.logical_and(fizz_v, buzz_v), tf.int32))
        return [fizz, buzz, fizzbuzz]

class FizzBuzz(tf.Module):
    def model(self,
              n  # Shape [] -- int32 the max number to loop FizzBuzz to
              ):  # Returns counts for fizz, buzz and fizzbuzz. Shape: [1] with length 3
        i = np.arange(n)
        fizz_v = i % 2 == 0
        buzz_v = i % 3 == 0
        fizz = np.sum(fizz_v.astype(np.int32))
        buzz = np.sum(buzz_v.astype(np.int32))
        fizzbuzz = np.sum(np.logical_and(fizz_v, buzz_v).astype(np.int32))
        return [fizz, buzz, fizzbuzz]

tf.xla.experimental.compile(fb_saved_model.model, [tf.constant(100000)])