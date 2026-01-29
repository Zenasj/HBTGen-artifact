# tf.random.normal((10, 9), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # As the original code uses tf.raw_ops.IgammaGradA inside the forward pass,
        # we will replicate that behavior here.
        # The param `a` is randomly generated inside the call with a fixed shape:
        # [1, 3, 6, 6, 1, 4, 6, 2, 1, 1]
        # The input `x` is expected with shape (10, 9)
        # Because tf.raw_ops.IgammaGradA is a low-level op related to incomplete gamma gradient,
        # no trainable variables or layers are used.

    @tf.function(jit_compile=True)
    def call(self, x):
        # According to source, igamma_grad_a accepts inputs:
        # x: Tensor, a: Tensor
        # Both must be broadcastable shapes; here 'a' is fixed shape, 'x' is input shape.
        a = tf.random.normal([1, 3, 6, 6, 1, 4, 6, 2, 1, 1], dtype=tf.float32)
        result = tf.raw_ops.IgammaGradA(x=x, a=a)
        return result


def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a tensor with shape and dtype matching expected input of MyModel
    # Input shape: (10, 9), dtype: float32 consistent with the original code
    return tf.random.normal([10, 9], dtype=tf.float32)

