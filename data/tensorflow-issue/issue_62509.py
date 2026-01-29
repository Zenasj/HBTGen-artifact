# tf.random.uniform((B, H, W, C, D), dtype=tf.float32) ← Assumed 5D input tensors based on concat and transpose operations along axis=4
import tensorflow as tf

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()
    @tf.function(jit_compile=True)
    def call(self, inp1, inp2):
        # Concatenate inp2, inp1 along axis 4 (5th dimension)
        conc = tf.concat([inp2, inp1], axis=4)
        # Reduce product along axis 4
        reduced = tf.math.reduce_prod(conc, axis=4)
        # Apply tan element-wise
        taned = tf.tan(reduced)
        return taned

class Model2(tf.keras.Model):
    def __init__(self):
        super().__init__()
    @tf.function(jit_compile=True)
    def call(self, inp1, inp2):
        # Transpose inp1 and inp2 permuting dims [4,1,2,3,0]
        transposed_inp1 = tf.transpose(inp1, perm=[4,1,2,3,0])
        transposed_inp2 = tf.transpose(inp2, perm=[4,1,2,3,0])
        # Concatenate transposed tensors along axis 0
        transposed_conc = tf.concat([transposed_inp2, transposed_inp1], axis=0)
        # Transpose back using the inverse of above permutation
        conc = tf.transpose(transposed_conc, perm=[4,1,2,3,0])
        # Reduce product along axis 4 again
        reduced = tf.math.reduce_prod(conc, axis=4)
        taned = tf.tan(reduced)
        return taned, conc

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Instantiate both sub-models
        self.model1 = Model1()
        self.model2 = Model2()

    @tf.function(jit_compile=True)
    def call(self, inp1, inp2):
        # Get output from model1 (single tensor)
        out1 = self.model1(inp1, inp2)
        # Get output and extra tensor from model2
        out2, conc = self.model2(inp1, inp2)

        # Compare the outputs from model1 and model2 with a tolerance
        # Using absolute difference and rtol, atol consistent with issue (rtol=0.001, atol=0.001)
        diff = tf.abs(out1 - out2)
        tol = 0.001 + 0.001 * tf.abs(out2)
        are_close = diff <= tol

        # Overall boolean indicating if output1 and output2 are close for all elements
        all_close = tf.reduce_all(are_close)

        # Return a dictionary-like structure of tensors:
        # - 'all_close' indicates if the outputs match under given tolerances
        # - 'diff' tensor showing absolute difference
        # - 'model1_out' tensor
        # - 'model2_out' tensor
        # - 'extra_conc' the extra concatenated tensor output from model2
        return {
            'all_close': all_close,
            'diff': diff,
            'model1_out': out1,
            'model2_out': out2,
            'extra_conc': conc,
        }

def my_model_function():
    # Return an instance of the fused MyModel class
    return MyModel()

def GetInput():
    # Based on operations done, inputs are 5D tensors with shape compatible for concat on axis=4
    # Assume shape: batch=2, height=3, width=3, channel=1, depth=4
    # Shapes of inp1 and inp2 must be same shape for tf.concat on axis=4
    shape = (2, 3, 3, 1, 4)
    inp1 = tf.random.uniform(shape, minval=0.1, maxval=1.0, dtype=tf.float32)
    inp2 = tf.random.uniform(shape, minval=0.1, maxval=1.0, dtype=tf.float32)
    return inp1, inp2

