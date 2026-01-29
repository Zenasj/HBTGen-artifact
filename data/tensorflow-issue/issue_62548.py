# tf.random.uniform((20, 20), dtype=tf.float64) ← Input shape and dtype inferred from issue repro code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Instantiate submodels corresponding to Model1 and Model2 from the issue
        self.model1 = Model1()
        self.model2 = Model2()

    def call(self, inp):
        """
        Run both Model1 and Model2 on input and compare their outputs.
        Return a dictionary with:
          - 'model1_output': the output from Model1 (single tensor)
          - 'model2_output': the first output from Model2 (tensor)
          - 'model2_concat': the second output from Model2 (tensor)
          - 'close': boolean tensor indicating elementwise close within tolerance between model1_output and model2_output
          - 'max_abs_diff': maximum absolute difference scalar between model1_output and model2_output
        """
        out1 = self.model1(inp)[0]  # Model1 returns tuple with one element
        out2, concat = self.model2(inp)  # Model2 returns tuple of two tensors (add, concat)
        
        # Compute elementwise closeness with tolerances matching the original np.testing.assert_allclose rtol=1e-3, atol=1e-3
        close = tf.math.abs(out1 - out2) <= (1e-3 + 1e-3 * tf.math.abs(out2))
        
        max_abs_diff = tf.reduce_max(tf.math.abs(out1 - out2))
        
        return {
            'model1_output': out1,
            'model2_output': out2,
            'model2_concat': concat,
            'close': close,
            'max_abs_diff': max_abs_diff
        }

# Reconstructed Model1 from issue
class Model1(tf.keras.Model):
    @tf.function(jit_compile=True)
    def call(self, inp):
        # transpose, floor, reverse, add (rev + floor)
        trans = tf.transpose(inp, perm=[1, 0])
        flo = tf.floor(trans)
        rev = tf.reverse(trans, axis=[0])
        add = tf.add(rev, flo)
        return (add,)

# Reconstructed Model2 from issue
class Model2(tf.keras.Model):
    @tf.function(jit_compile=True)
    def call(self, inp):
        # concat input with itself, transpose, floor, reverse, add (floor + rev)
        concat = tf.concat([inp, inp], axis=0)
        trans = tf.transpose(inp, perm=[1, 0])
        flo = tf.floor(trans)
        rev = tf.reverse(trans, axis=[0])
        add = tf.add(flo, rev)
        return (add, concat)

def my_model_function():
    # Return an instance of fused MyModel encapsulating Model1 and Model2
    return MyModel()

def GetInput():
    # Provide a random tensor matching input shape [20,20] and dtype float64 per original code
    # Use seed for reproducibility
    tf.random.set_seed(42)
    return tf.random.uniform(shape=[20, 20], dtype=tf.float64)

