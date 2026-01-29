# tf.random.uniform((1, 40, 1, 31, 49), dtype=tf.float64)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No extra layers needed; operations are functional/transforms

    @tf.function(jit_compile=True)
    def call(self, inp):
        # Model1 computation
        trans = tf.transpose(inp, perm=[4, 1, 2, 3, 0])  # shape: [49,40,1,31,1]
        rev = tf.reverse(trans, axis=[0, 2, 3, 4])       # reverse over several axes
        add = tf.add(rev, trans)                         # elementwise add

        split1_m1, split2_m1 = tf.split(add, 2, axis=1) # split along axis=1 (dim 40 -> 2 x 20)

        # Model2 computation
        trans_output = tf.transpose(tf.concat([trans, trans], axis=0), perm=[1, 0, 2, 3, 4])
        # concat along axis=0 doubles dim 0 from 49 to 98, then permutes back
        rev2 = tf.reverse(trans, axis=[0, 2, 3, 4])
        add2 = tf.add(trans, rev2)
        split1_m2, split2_m2 = tf.split(add2, 2, axis=1)

        # The original issue highlighted numeric mismatches between split outputs of Model1 and Model2.
        # We return a combined comparison output showing if all splits match within a tolerance.

        # Because outputs are float64, use tf.math.abs and tolerance manual check
        rtol = 0.001
        atol = 0.001

        def close_enough(a, b):
            diff = tf.math.abs(a - b)
            tol = atol + rtol * tf.math.abs(b)
            return diff <= tol

        # Comparisons for splits from Model1 vs Model2
        comp_split1 = tf.reduce_all(close_enough(split1_m1, split1_m2))
        comp_split2 = tf.reduce_all(close_enough(split2_m1, split2_m2))

        # Output a dict-like structure as a tuple: (comparison results, and the trans_output from Model2)
        # This encapsulates both models and their comparison as requested.
        # Converting booleans to tf.bool tensors for consistent output.
        out = (comp_split1, comp_split2, trans_output)

        return out

def my_model_function():
    return MyModel()

def GetInput():
    # Based on the input shape from example: [1, 40, 1, 31, 49], float64 dtype
    return tf.random.uniform(shape=(1, 40, 1, 31, 49), dtype=tf.float64)

