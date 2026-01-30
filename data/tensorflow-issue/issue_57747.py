import tensorflow as tf
print(tf.__version__)
from keras import layers


class MyModule(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def __call__(self, x):
        y = tf.divide(x, x)
        z = tf.pow(x, y)
        return z



sys_details = tf.sysconfig.get_build_info()
print(f'cuda_version: {sys_details["cuda_version"]}')
print(f'cudnn_version: {sys_details["cudnn_version"]}')
print(f'cuda_compute_capabilities: {sys_details["cuda_compute_capabilities"]}')

m = MyModule()
x = tf.constant(
    [-0.1982182], dtype=tf.float32,
)
with tf.device('/CPU:0'):
    tf.config.run_functions_eagerly(True)
    out = m(x)
    print(out) # NOTE: RIGHT! tf.Tensor([-0.1982182], shape=(1,), dtype=float32)
    tf.config.run_functions_eagerly(False)

with tf.device('/CPU:0'):
    out = m(x)
    print(out) # NOTE: RIGHT! tf.Tensor([-0.1982182], shape=(1,), dtype=float32)

with tf.device('/GPU:0'): # NOTE: GPU needed!
    out = m(x)
    print(out) # NOTE: WRONG! tf.Tensor([nan], shape=(1,), dtype=float32)

import tensorflow as tf

@tf.function(jit_compile=True)
def div(x, y):
    return tf.divide(x, y)

x = tf.constant([-0.1982182], dtype=tf.float32)

with tf.device('/CPU:0'):
    print( div(x, x) ) # tf.Tensor([1.], shape=(1,), dtype=float32)

with tf.device('/GPU:0'):
    print( div(x, x) ) # tf.Tensor([0.99999994], shape=(1,), dtype=float32) <-- imprecise result