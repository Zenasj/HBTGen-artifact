import math
import tensorflow as tf

tf.math.real(tf.convert_to_tensor(something))
tf.math.real(something)

real_python = tf.math.real(1.)  # <- fails

from tensorflow.python import ops

class MyTensor():
    def _dense_var_to_tensor(self, dtype, name, as_ref):
        return tf.constant(42, dtype=dtype)


def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
    return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)


ops.register_tensor_conversion_function(MyTensor, _dense_var_to_tensor)

my_t1 = MyTensor()
t1 = tf.convert_to_tensor(my_t1)  # <- works
square = tf.math.round(my_t1)  # <- works
real_python = tf.math.real(1.)  # <- fails
real = tf.math.real(my_t1)  # <- fails