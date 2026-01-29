# tf.random.uniform((64, 64), dtype=tf.float64) and tf.random.uniform((1, 1, 64), dtype=tf.int32)
import tensorflow as tf

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()
    
    @tf.function(jit_compile=True)
    def call(self, inp1, inp2):
        # inp1: [64, 64], dtype=float64 (using float64 since TF does not have complex128 support well for ops like multiply, 
        # original used float64 in snippet to simulate complex128)
        # inp2: [1, 1, 64], dtype=int32
        
        # Transpose inp1 from [64, 64] to [64, 64] but permuted axes (perm=[1, 0])
        trans = tf.transpose(inp1, perm=[1, 0])
        # Gather indices clipped to [0,63] along axis 0
        gather = tf.gather(trans, tf.clip_by_value(inp2, 0, 63), axis=0)
        # Squeeze axis 1 to reduce dimensions from [?, 1, ?] to [?, ?]
        squeeze = tf.squeeze(gather, axis=1)
        # Multiply squeeze element-wise by itself
        mul1 = tf.multiply(squeeze, squeeze)
        # Multiply trans by mul1 - the original Model1 expression
        mul2 = tf.multiply(trans, mul1)
        return mul2

class Model2(tf.keras.Model):
    def __init__(self):
        super().__init__()
    
    @tf.function(jit_compile=True)
    def call(self, inp1, inp2):
        # Same input shapes and types as Model1
        
        trans = tf.transpose(inp1, perm=[1, 0])
        gather = tf.gather(trans, tf.clip_by_value(inp2, 0, 63), axis=0)
        squeeze = tf.squeeze(gather, axis=1)
        mul1 = tf.multiply(squeeze, squeeze)
        # Different order of multiplication compared to Model1
        mul2 = tf.multiply(mul1, trans)
        return mul2

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Instantiate the two models
        self.model1 = Model1()
        self.model2 = Model2()
    
    @tf.function(jit_compile=True)
    def call(self, inp1, inp2):
        # Run both sub-models with the same inputs
        out1 = self.model1(inp1, inp2)
        out2 = self.model2(inp1, inp2)
        
        # Compare outputs element-wise within a small tolerance
        # To replicate typical np.testing.assert_allclose logic,
        # compute a boolean tensor indicating if within tolerance
        
        atol = 1e-3
        rtol = 1e-3
        
        # Absolute difference
        abs_diff = tf.abs(out1 - out2)
        # Relative difference
        rel_diff = abs_diff / tf.maximum(tf.abs(out2), 1e-12)  # avoid div by zero
        
        # Boolean mask of where outputs are close enough
        close = tf.logical_or(abs_diff <= atol, rel_diff <= rtol)
        
        # Final result: True if all elements close, else False
        all_close = tf.reduce_all(close)
        
        # For more info, return a tuple of:
        # (all_close boolean, difference tensor, outputs from both models)
        # but per requirement output being the comparison, we return a dict
        
        # We pack outputs in a dict for clarity, or just return a tuple
        return {
            "all_close": all_close,
            "diff_absolute": abs_diff,
            "model1_output": out1,
            "model2_output": out2,
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create inputs exactly matching the original shapes and dtypes in the issue:
    # inp1: [64,64], dtype float64 (using uniform in [0,1])
    # inp2: [1,1,64], dtype int32, values ranging from -100 to 100
    
    inp1 = tf.random.uniform([64, 64], dtype=tf.float64)
    inp2 = tf.random.uniform([1, 1, 64], minval=-100, maxval=100, dtype=tf.int32)
    return inp1, inp2

