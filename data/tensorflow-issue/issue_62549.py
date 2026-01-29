# tf.random.uniform((16, 16), dtype=tf.float64) ← Input is a 2D square matrix of shape [16,16] with float64 dtype
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define sub-models reflecting Model1 and Model2 behavior
        
        # Model1: add = tf.add(rev, subtract)
        # Model2: add = tf.add(subtract, rev)
        # Here rev = reverse(transpose), subtract = trans - trans (zero tensor)
        # The bug appears when operands to tf.add are swapped under XLA on GPU.
        # We encapsulate both and produce their outputs, then compare.

    @tf.function(jit_compile=True)
    def call(self, inp):
        trans = tf.transpose(inp, perm=[1, 0])
        rev = tf.reverse(trans, axis=[0, 1])
        subtract = tf.math.subtract(trans, trans)

        # Model1 computation: add rev + subtract
        add1 = tf.add(rev, subtract)

        # Model2 computation: add subtract + rev
        add2 = tf.add(subtract, rev)

        # Produce boolean tensor indicating element-wise closeness within tolerance:
        # This reflects whether the results match despite operand order swap
        close = tf.math.abs(add1 - add2) <= (0.001 + 0.001 * tf.math.abs(add2))

        # Additionally return numerical difference for diagnostic purposes
        diff = add1 - add2

        # Return tuple with both model outputs, boolean close tensor, and numeric diff
        # This offers comprehensive insight into discrepancy under XLA GPU compilation
        return add1, add2, close, diff


def my_model_function():
    # Instantiate MyModel without any external weights or configs
    return MyModel()


def GetInput():
    # Return a tensor matching input expected by MyModel: shape [16, 16], dtype float64,
    # random uniform values in [0,1)
    return tf.random.uniform(shape=[16, 16], dtype=tf.float64)

