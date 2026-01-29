# tf.random.uniform((3,), dtype=tf.int32), tf.random.uniform((), dtype=tf.int32), tf.random.uniform((3,), dtype=tf.int32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Encapsulate the two model variants as submodules
        # Model1: mul3 = (inp3 * (inp2*inp1)) * (inp3 * (inp2*inp1))
        # Model2: mul3 = inp3 * ((inp1*inp2) * (inp3 * (inp1*inp2)))
        # Both return mul3 and tf.abs(mul3)
    
    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2, inp3):
        # Compute Model1 output
        mul1_left = tf.multiply(inp1, inp2)
        mul1_right = tf.multiply(inp3, mul1_left)
        mul1 = tf.multiply(mul1_right, mul1_right)
        abs1 = tf.abs(mul1)

        # Compute Model2 output
        mul2_left = tf.multiply(inp1, inp2)
        mul2_right = tf.multiply(inp3, mul2_left)
        mul2 = tf.multiply(inp3, tf.multiply(mul2_left, mul2_right))
        abs2 = tf.abs(mul2)

        # Compare outputs: return boolean mask where abs1 and abs2 are close within tolerance
        # This highlights discrepancy shown in the issue
        close_mul = tf.experimental.numpy.isclose(mul1, mul2, rtol=1e-3, atol=1e-3)
        close_abs = tf.experimental.numpy.isclose(abs1, abs2, rtol=1e-3, atol=1e-3)

        # Return the model outputs plus comparison masks
        # Shape is preserved from mul1/mul2 which is same as input shapes
        return {
            'mul_model1': mul1,
            'abs_model1': abs1,
            'mul_model2': mul2,
            'abs_model2': abs2,
            'close_mul': close_mul,
            'close_abs': close_abs
        }

def my_model_function():
    # Return instance of MyModel with no weights or init params
    return MyModel()

def GetInput():
    # Generate input tuple (inp1, inp2, inp3) matching shapes and dtypes expected by MyModel
    # Based on issue:
    # inp1: tensor shape [3], dtype tf.int32 uniformly sampled [-100, 100)
    # inp2: scalar tensor (), dtype tf.int32 uniformly sampled [-100, 100)
    # inp3: tensor shape [3], dtype tf.int32 uniformly sampled [-100, 100)
    inp1 = tf.random.uniform(shape=[3], minval=-100, maxval=100, dtype=tf.int32)
    inp2 = tf.random.uniform(shape=[], minval=-100, maxval=100, dtype=tf.int32)
    inp3 = tf.random.uniform(shape=[3], minval=-100, maxval=100, dtype=tf.int32)
    return (inp1, inp2, inp3)

