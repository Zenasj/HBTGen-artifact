import tensorflow as tf
print(tf.__version__)
from keras import layers


class MyModule(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
        x = tf.sqrt(x)
        x = tf.raw_ops.LRN(
            input=x,
            depth_radius=1,
            bias=1,
            alpha=1,
            beta=1,
        ) # input / (bias + alpha * sqr_sum) ** beta
        return x


m = MyModule()
x = tf.constant(
    [[[[-0.5,  0.5,  0.5]]]], dtype=tf.float32,
)
with tf.device('/CPU:0'):
    tf.config.run_functions_eagerly(True)
    out = m(x)
    print(out) # tf.Tensor([[[[nan nan nan]]]], shape=(1, 1, 1, 3), dtype=float32)
    tf.config.run_functions_eagerly(False)

with tf.device('/CPU:0'):
    out = m(x)
    print(out) # NOTE: WRONG! tf.Tensor([[[[       nan        nan 0.35355338]]]], shape=(1, 1, 1, 3), dtype=float32)