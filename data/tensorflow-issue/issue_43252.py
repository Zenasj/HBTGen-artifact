import random

import os
import tensorflow as tf
import time

class Issue_fwd(tf.Module):

    @tf.function(input_signature=[tf.TensorSpec([None, 1], tf.float64)] * 3 +
                                 [tf.TensorSpec([None, 3], tf.float64)] +
                                 [tf.TensorSpec([1, None], tf.float64)] * 4)
    def f(self, x1, x2, x3, c, v1, v2, v3, v4):

        with tf.autodiff.ForwardAccumulator(x1, tf.ones_like(x1)) as fwd_acc_x1_2, \
                tf.autodiff.ForwardAccumulator(x2, tf.ones_like(x2)) as fwd_acc_x2_2:

            with tf.autodiff.ForwardAccumulator(x1, tf.ones_like(x1)) as fwd_acc_x1, \
                 tf.autodiff.ForwardAccumulator(x2, tf.ones_like(x2)) as fwd_acc_x2, \
                 tf.autodiff.ForwardAccumulator(x3, tf.ones_like(x3)) as fwd_acc_x3:

                p = tf.concat([x1, x2, x3], axis=1)
                pe = tf.transpose(a=p[:, :, None], perm=[0, 2, 1])
                ce = tf.transpose(a=c[:, :, None], perm=[2, 0, 1])
                r = tf.reduce_sum(input_tensor=tf.square(ce - pe), axis=2)
                G = tf.exp(-r / 2)

                p = tf.reduce_sum(input_tensor=G * v1, axis=1, keepdims=True)
                b = tf.reduce_sum(input_tensor=G * v2, axis=1, keepdims=True)
                u = tf.reduce_sum(input_tensor=G * v3, axis=1, keepdims=True)
                w = tf.reduce_sum(input_tensor=G * v4, axis=1, keepdims=True)

            dpdx = fwd_acc_x1.jvp(p, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dbdx = fwd_acc_x1.jvp(b, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dudx = fwd_acc_x1.jvp(u, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dwdx = fwd_acc_x1.jvp(w, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            dpdz = fwd_acc_x2.jvp(p, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dbdz = fwd_acc_x2.jvp(b, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dudz = fwd_acc_x2.jvp(u, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dwdz = fwd_acc_x2.jvp(w, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            dbdt = fwd_acc_x3.jvp(b, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dudt = fwd_acc_x3.jvp(u, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dwdt = fwd_acc_x3.jvp(w, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        d2ud2x = fwd_acc_x1_2.jvp(dudx, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2ud2z = fwd_acc_x2_2.jvp(dudz, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        d2wd2x = fwd_acc_x1_2.jvp(dwdx, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2wd2z = fwd_acc_x2_2.jvp(dwdz, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        d2bd2x = fwd_acc_x1_2.jvp(dbdx, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2bd2z = fwd_acc_x2_2.jvp(dbdz, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        return dudx, dudz, dudt, dwdx, dwdz, dwdt, dbdx, dbdz, dbdt, dpdx, dpdz,  d2ud2x, d2ud2z, d2wd2x, d2wd2z, d2bd2x, d2bd2z,


f = Issue_fwd()
saving_path = 'save_path'
os.makedirs(saving_path, exist_ok=True)

start_time = time.clock()
tf.saved_model.save(f, saving_path)
delta_time = time.clock() - start_time
print('saving took {:f} seconds'.format(delta_time))
print('tf.version.GIT_VERSION={}'.format(tf.version.GIT_VERSION))
print('tf.version.VERSION={}'.format(tf.version.VERSION))

import tensorflow as tf
import time

class Issue_fwd(tf.Module):

    input_signature=([tf.TensorSpec([None, 1], tf.float64)] * 3 +
                                 [tf.TensorSpec([None, 3], tf.float64)] +
                                 [tf.TensorSpec([1, None], tf.float64)] * 4)
    @tf.function
    def f(self, x1, x2, x3, c, v1, v2, v3, v4):

        with tf.autodiff.ForwardAccumulator(x1, tf.ones_like(x1)) as fwd_acc_x1_2, \
                tf.autodiff.ForwardAccumulator(x2, tf.ones_like(x2)) as fwd_acc_x2_2:

            with tf.autodiff.ForwardAccumulator(x1, tf.ones_like(x1)) as fwd_acc_x1, \
                 tf.autodiff.ForwardAccumulator(x2, tf.ones_like(x2)) as fwd_acc_x2, \
                 tf.autodiff.ForwardAccumulator(x3, tf.ones_like(x3)) as fwd_acc_x3:

                p = tf.concat([x1, x2, x3], axis=1)
                pe = tf.transpose(a=p[:, :, None], perm=[0, 2, 1])
                ce = tf.transpose(a=c[:, :, None], perm=[2, 0, 1])
                r = tf.reduce_sum(input_tensor=tf.square(ce - pe), axis=2)
                G = tf.exp(-r / 2)

                p = tf.reduce_sum(input_tensor=G * v1, axis=1, keepdims=True)
                b = tf.reduce_sum(input_tensor=G * v2, axis=1, keepdims=True)
                u = tf.reduce_sum(input_tensor=G * v3, axis=1, keepdims=True)
                w = tf.reduce_sum(input_tensor=G * v4, axis=1, keepdims=True)

            dpdx = fwd_acc_x1.jvp(p, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dbdx = fwd_acc_x1.jvp(b, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dudx = fwd_acc_x1.jvp(u, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dwdx = fwd_acc_x1.jvp(w, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            dpdz = fwd_acc_x2.jvp(p, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dbdz = fwd_acc_x2.jvp(b, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dudz = fwd_acc_x2.jvp(u, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dwdz = fwd_acc_x2.jvp(w, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            dbdt = fwd_acc_x3.jvp(b, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dudt = fwd_acc_x3.jvp(u, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            dwdt = fwd_acc_x3.jvp(w, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        d2ud2x = fwd_acc_x1_2.jvp(dudx, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2ud2z = fwd_acc_x2_2.jvp(dudz, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        d2wd2x = fwd_acc_x1_2.jvp(dwdx, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2wd2z = fwd_acc_x2_2.jvp(dwdz, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        d2bd2x = fwd_acc_x1_2.jvp(dbdx, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2bd2z = fwd_acc_x2_2.jvp(dbdz, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        return dudx, dudz, dudt, dwdx, dwdz, dwdt, dbdx, dbdz, dbdt, dpdx, dpdz,  d2ud2x, d2ud2z, d2wd2x, d2wd2z, d2bd2x, d2bd2z,


issue_fwd = Issue_fwd()

n = 10
x1 = tf.random.uniform((n, 1), dtype=tf.float64)
x2 = tf.random.uniform((n, 1), dtype=tf.float64)
x3 = tf.random.uniform((n, 1), dtype=tf.float64)

c = tf.random.uniform((n,3), dtype=tf.float64)

v1 = tf.random.uniform((1, n), dtype=tf.float64)
v2 = tf.random.uniform((1, n), dtype=tf.float64)
v3 = tf.random.uniform((1, n), dtype=tf.float64)
v4 = tf.random.uniform((1, n), dtype=tf.float64)


start_time = time.clock()
dudx, dudz, dudt, dwdx, dwdz, dwdt, dbdx, dbdz, dbdt, dpdx, dpdz, d2ud2x, d2ud2z, d2wd2x, d2wd2z, d2bd2x, d2bd2z = issue_fwd.f(x1, x2, x3, c, v1, v2, v3, v4)
delta_time = time.clock() - start_time

print('running took {:f} seconds'.format(delta_time))
print('tf.version.GIT_VERSION={}'.format(tf.version.GIT_VERSION))
print('tf.version.VERSION={}'.format(tf.version.VERSION))

import os
import tensorflow as tf
import time


def _jvp(f, primals, tangents):
    with tf.autodiff.ForwardAccumulator(primals, tangents) as acc:
        primals_out = f(*primals)
    return primals_out, acc.jvp(
        primals_out, unconnected_gradients=tf.UnconnectedGradients.ZERO)


input_signature = [tf.TensorSpec([None, 1], tf.float64)] * 3 + \
                     [tf.TensorSpec([None, 3], tf.float64)] + \
                     [tf.TensorSpec([1, None], tf.float64)] * 4

@tf.function(input_signature=input_signature)
def ff(x, z, t, c, v1, v2, v3, v4):

    p = tf.concat([x, z, t], axis=1)
    pe = tf.transpose(a=p[:, :, None], perm=[0, 2, 1])
    ce = tf.transpose(a=c[:, :, None], perm=[2, 0, 1])
    d = ce - pe
    r = tf.reduce_sum(input_tensor=tf.square(d), axis=2)
    G = tf.exp(-r / 2)

    p = tf.reduce_sum(input_tensor=G * v1, axis=1, keepdims=True)
    b = tf.reduce_sum(input_tensor=G * v2, axis=1, keepdims=True)
    u = tf.reduce_sum(input_tensor=G * v3, axis=1, keepdims=True)
    w = tf.reduce_sum(input_tensor=G * v4, axis=1, keepdims=True)

    return p, b, u, w

class Issue_fwd(tf.Module):
    @tf.function(input_signature=input_signature)
    def f(self, x, z, t, c, v1, v2, v3, v4):

        fi = lambda xi, zi, ti: ff(xi, zi, ti, c, v1, v2, v3, v4)
        primals = [x, z, t]
        tangent_mask = [tf.zeros_like(primal) for primal in primals]

        with tf.autodiff.ForwardAccumulator(primals=[x], tangents=[tf.ones_like(x)]) as fwd_outer:
            i = 0
            primals = [x, z, t]
            [dpdx, dbdx, dudx, dwdx] = _jvp(fi, primals, tangent_mask[:i] + [tf.ones_like(primals[i])] + tangent_mask[i + 1:])[1]
        [d2bd2x, d2ud2x, d2wd2x] = fwd_outer.jvp([dbdx, dudx, dwdx], tf.UnconnectedGradients.ZERO)

        with tf.autodiff.ForwardAccumulator(primals=[z], tangents=[tf.ones_like(z)]) as fwd_outer:
            i = 1
            primals = [x, z, t]
            [dpdz, dbdz, dudz, dwdz] = _jvp(fi, primals, tangent_mask[:i] + [tf.ones_like(primals[i])] + tangent_mask[i + 1:])[1]
        [d2bd2z, d2ud2z, d2wd2z] = fwd_outer.jvp([dbdz, dudz, dwdz], tf.UnconnectedGradients.ZERO)

        i = 2
        [p, b, u, w], [dpdt, dbdt, dudt, dwdt] = _jvp(fi, primals, tangent_mask[:i] + [tf.ones_like(primals[i])] + tangent_mask[i + 1:])

        return dudx, dudz, dudt, dwdx, dwdz, dwdt, dbdx, dbdz, dbdt, dpdx, dpdz, d2ud2x, d2ud2z, d2wd2x, d2wd2z, d2bd2x, d2bd2z,


issue_fwd = Issue_fwd()
saving_path = 'save_path'
os.makedirs(saving_path, exist_ok=True)

start_time = time.clock()
tf.saved_model.save(issue_fwd, saving_path)
delta_time = time.clock() - start_time
print('saving took {:f} seconds'.format(delta_time))
print('tf.version.GIT_VERSION={}'.format(tf.version.GIT_VERSION))
print('tf.version.VERSION={}'.format(tf.version.VERSION))


n = 5000
x = tf.random.uniform((n, 1), dtype=tf.float64)
z = tf.random.uniform((n, 1), dtype=tf.float64)
t = tf.random.uniform((n, 1), dtype=tf.float64)

c = tf.random.uniform((n,3), dtype=tf.float64)

v1 = tf.random.uniform((1, n), dtype=tf.float64)
v2 = tf.random.uniform((1, n), dtype=tf.float64)
v3 = tf.random.uniform((1, n), dtype=tf.float64)
v4 = tf.random.uniform((1, n), dtype=tf.float64)


start_time = time.clock()
p, b, u, w = ff(x, z, t, c, v1, v2, v3, v4)
delta_time = time.clock() - start_time
print('running ff took {:f} seconds'.format(delta_time))



start_time = time.clock()
dudx, dudz, dudt, dwdx, dwdz, dwdt, dbdx, dbdz, dbdt, dpdx, dpdz, d2ud2x, d2ud2z, d2wd2x, d2wd2z, d2bd2x, d2bd2z = issue_fwd.f(x, z, t, c, v1, v2, v3, v4)
delta_time = time.clock() - start_time
print('running Issue_fwd.f took {:f} seconds'.format(delta_time))