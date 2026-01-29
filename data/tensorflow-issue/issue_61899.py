# tf.constant shape=[1, 2], dtype=tf.float32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def call(self, x):
        # Compute top_k with k=2 on the input tensor x, unsorted
        values, indices = tf.math.top_k(x, k=2, sorted=False)
        # Slice values with size=[0,1] - results in a zero-size slice tensor
        # This pattern reproduces the issue where TFLite fails on zero-size slice,
        # but original TF supports it.
        y = tf.slice(values, tf.constant([0, 0]), tf.constant([0, 1]))
        return y

def my_model_function():
    # Returns an instance of MyModel with no special initialization or weights
    return MyModel()

def GetInput():
    # Return a input tensor matching the expected shape of MyModel input
    # Here input shape is [1, 2] with float32 dtype, as seen in the example
    # The values [1., 2.] are chosen to match example, but random inputs also work
    return tf.constant([1., 2.], shape=[1, 2], dtype=tf.float32)

