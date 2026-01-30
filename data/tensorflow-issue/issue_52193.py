import tensorflow as tf
import numpy as np

class TensorWrapper:
    def __init__(self, tensor):
        self._tensor = tensor  # suppose we have no way to directly access the tensor

    def plus(self, other):
        # only public API
        return other + self._tensor

    def __str__(self):
        return "TensorWrapper(" + self._tensor.__str__() + ")"

    __repr__ = __str__


TW_list_tf = list(map(TensorWrapper, [tf.zeros([2, 2]), tf.ones([2, 2])]))

print(TW_list_tf)


def u_tf(a, b):
    return tf.reduce_sum(TW_list_tf[b].plus(a))

print("u_tf ", u_tf(tf.zeros([2, 2]), 1))

u_tf_jit = tf.function(u_tf)

print("u_tf_jit", u_tf_jit(tf.zeros([2, 2]), 1))

a = tf.zeros([2, 2])
with tf.GradientTape() as tape:
    tape.watch(a)
    loss = u_tf(a, 1)
print("u_tf grad ", tape.gradient(loss, a))

a = tf.zeros([2, 2])
with tf.GradientTape() as tape:
    tape.watch(a)
    loss = u_tf_jit(a, 1)
print("u_tf_jit grad ", loss, tape.gradient(loss, a))

@tf.custom_gradient
def u_tf_grad_v2(a, b):
    r = u_tf(a, b)

    def grad(dr):
        return 2.0 * dr * tf.ones_like(a), tf.zeros_like(b)

    return r, grad

a = tf.zeros([2, 2])
with tf.GradientTape() as tape:
    tape.watch(a)
    loss = u_tf_grad_v2(a, 1)
print("u_tf_grad_v2 grad ", tape.gradient(loss, a))

u_tf_grad_v2_jit = tf.function(u_tf_grad_v2)

try:
    a = tf.zeros([2, 2])
    with tf.GradientTape() as tape:
        tape.watch(a)
        loss = u_tf_grad_v2_jit(a, 1)
    print(loss, tape.gradient(loss, a))
except Exception as e:
    print("u_tf_grad_v2_jit grad:", e)
# list indices must be integers or slices, not Tensor since int b is compiled to tensor