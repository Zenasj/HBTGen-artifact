# tf.random.uniform((1, 1, 1, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def call(self, x):
        # This model applies sqrt followed by Local Response Normalization (LRN).
        # According to the issue, with XLA (jit_compile=True), LRN returns inconsistent values
        # when input contains NaNs: some outputs should be NaN but are not.
        # The model reproduces that behavior by invoking tf.raw_ops.LRN.
        x = tf.sqrt(x)
        x = tf.raw_ops.LRN(
            input=x,
            depth_radius=1,
            bias=1,
            alpha=1,
            beta=1,
        )
        return x

def my_model_function():
    # Returns an instance of MyModel, no weights to initialize.
    return MyModel()

def GetInput():
    # Generate the input tensor used in the issue, shape (1,1,1,3)
    # The input contains negative and positive values to produce NaNs after sqrt.
    # Specifically, sqrt(-0.5) should produce nan, sqrt(0.5) valid.
    return tf.constant(
        [[[[-0.5, 0.5, 0.5]]]],
        dtype=tf.float32,
    )

