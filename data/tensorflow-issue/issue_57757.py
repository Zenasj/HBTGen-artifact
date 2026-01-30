import tensorflow as tf
print(tf.__version__)
from keras import layers


class Network(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
        y0 = tf.raw_ops.Softmax(logits=x)
        y1 = tf.sqrt(x)
        y2 = tf.pow(y0, y1)
        return y2


net = Network()
x = tf.constant(
    [-0.3523314], dtype=tf.float32,
)

tf.config.run_functions_eagerly(True)
res = net(x)
print(res)
# tf.Tensor([1.], shape=(1,), dtype=float32)
tf.config.run_functions_eagerly(False)

res = net(x)
print(res)
# tf.Tensor([nan], shape=(1,), dtype=float32)