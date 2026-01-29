# tf.random.normal((6, 10, 1, 1), dtype=tf.bfloat16)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, divisor_tensor):
        super().__init__()
        # Store a fixed divisor tensor to avoid random differences inside the call method
        # This matches the recommendation to generate the divisor outside the call to avoid XLA random seed issues
        self.divisor_tensor = tf.Variable(divisor_tensor, trainable=False, dtype=tf.bfloat16)

    @tf.function(jit_compile=True)
    def call(self, x):
        # Use tf.raw_ops.DivNoNan operation
        # Divides x by divisor_tensor, with no NaNs if divisor_tensor contains zeros
        return tf.raw_ops.DivNoNan(x=x, y=self.divisor_tensor)

def my_model_function():
    # Create a fixed divisor_tensor input outside the model to ensure reproducibility with jit_compile
    divisor_tensor = tf.random.normal([6, 10, 1, 1], dtype=tf.bfloat16)
    return MyModel(divisor_tensor)

def GetInput():
    # Input 'x' must be compatible with tf.raw_ops.DivNoNan and divisor_tensor shape broadcast rules
    # We assume 'x' shape is broadcastable to divisor_tensor shape, so shape [1, 1], dtype bfloat16 is used as in the issue
    return tf.random.normal([1, 1], dtype=tf.bfloat16)

