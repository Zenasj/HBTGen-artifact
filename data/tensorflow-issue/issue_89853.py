import tensorflow as tf
class DummyTensor:
    def __init__(self, data):
        self.data = data
    def __tf_function__(self, func, types, args=(), kwargs=None):
        if func is tf.compat.v2.nn.depthwise_conv2d_backprop_input:
            return tf.constant(-1.0, dtype=tf.float32)
        return NotImplemented

x = DummyTensor(None)
input_sizes = tf.constant([1, 4, 4, 1], dtype=tf.int32)
filter_tensor = tf.constant([[1.0]], dtype=tf.float32)
out_backprop = tf.constant([[1.0]], dtype=tf.float32)
result = tf.compat.v2.nn.depthwise_conv2d_backprop_input(input_sizes, filter_tensor, out_backprop, strides=[1, 1], padding='VALID')