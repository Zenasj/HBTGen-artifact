# tf.complex(tf.random.uniform((45, 29), dtype=tf.float64), tf.random.uniform((45, 29), dtype=tf.float64))
import tensorflow as tf

# Shared Dense layer used by both submodels
densenet = tf.keras.layers.Dense(units=1, dtype=tf.float32, autocast=False)

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = densenet

    @tf.function(jit_compile=True)
    def call(self, inp):
        cast = tf.cast(inp, dtype=tf.float32)
        dense = self.dense(cast)
        add = tf.add(dense, cast)
        reduce_min = tf.math.reduce_min(add, axis=1)
        return reduce_min,

class Model2(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = densenet

    @tf.function(jit_compile=True)
    def call(self, inp):
        cast = tf.cast(inp, dtype=tf.float32)
        ceil = tf.math.ceil(cast)
        dense = self.dense(cast)
        add = tf.add(cast, dense)
        trans1 = tf.transpose(add, perm=[1, 0])
        reduce_min = tf.math.reduce_min(add, axis=1)
        return reduce_min, ceil, trans1

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model1 = Model1()
        self.model2 = Model2()
        
    @tf.function(jit_compile=True)
    def call(self, inp):
        # Run both models' forward pass
        out1 = self.model1(inp)   # tuple with single element (reduce_min,)
        out2 = self.model2(inp)   # tuple with three elements (reduce_min, ceil, trans1)
        
        # Comparison logic:
        # Compare the first output tensors of model1 and model2 (the reduce_min output)
        # Use absolute tolerance 0.001 and relative tolerance 0.001 to emulate the original np.testing.assert_allclose
        
        m1_out = out1[0]
        m2_out = out2[0]
        
        # Compute difference and tolerance mask
        abs_diff = tf.abs(m1_out - m2_out)
        tol = 0.001 + 0.001 * tf.abs(m2_out)
        close = abs_diff <= tol
        
        # Return a boolean tensor indicating if all elements are close, plus actual numerical diffs
        all_close = tf.reduce_all(close)
        diff_summary = {
            'all_close': all_close,
            'max_abs_diff': tf.reduce_max(abs_diff),
            'max_rel_diff': tf.reduce_max(abs_diff / (tf.abs(m2_out) + 1e-10))
        }
        # Pack output as dictionary along with other outputs for user's reference
        # For ease of use, return a dictionary with keys
        return {
            'model1_reduce_min': m1_out,
            'model2_reduce_min': m2_out,
            'model2_ceil': out2[1],
            'model2_transpose': out2[2],
            'comparison': diff_summary,
        }

def my_model_function():
    # Returns an instance of the fused MyModel
    return MyModel()

def GetInput():
    # Returns the input tensor expected by MyModel
    # Input shape = (45, 29) complex128 tensor (dtype=tf.complex128)
    shape = (45, 29)
    real_part = tf.random.uniform(shape, dtype=tf.float64)
    imag_part = tf.random.uniform(shape, dtype=tf.float64)
    complex_input = tf.complex(real_part, imag_part)
    return complex_input

