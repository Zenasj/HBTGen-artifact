import math
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

class CONV_OP(tf.keras.layers.Layer):
    def __init__(self, n_f=32, ifactivate=False):
        super(CONV_OP, self).__init__()
        self.mylayers = []
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        if ifactivate == True:
            self.mylayers.append(tf.keras.layers.ReLU())
        self.seq = tf.keras.Sequential(self.mylayers)

    def call(self, input):
        res = self.seq(input)
        return res


class exam_net(tf.keras.Model):
    def __init__(self):
        super(exam_net, self).__init__(name='exam_net')
        self.conv = CONV_OP(n_f=2, ifactivate=False)
        self.conv_t = CONV_OP(n_f=2, ifactivate=False)
        self.thres_coef = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=True, name='thres_coef')

    def call(self, x_in):
        [batch, Nt, Nx, Ny] = x_in.get_shape()

        x_in = tf.stack([tf.math.real(x_in), tf.math.imag(x_in)], axis=-1)
        x = self.conv(x_in)

        x_c = tf.complex(x[:, :, :, :, 0], x[:, :, :, :, 1])

        St, Ut, Vt = tf.linalg.svd(x_c, compute_uv=True, full_matrices=True)
        thres = tf.sigmoid(self.thres_coef) * St[..., 0]
        thres = tf.expand_dims(thres, -1)
        St = tf.nn.relu(St - thres)
        St = tf.linalg.diag(St)

        St = tf.dtypes.cast(St, tf.complex64)
        Vt_conj = tf.transpose(Vt, perm=[0, 1, 3, 2])
        Vt_conj = tf.math.conj(Vt_conj)
        US = tf.linalg.matmul(Ut, St)
        x_soft = tf.linalg.matmul(US, Vt_conj)
        
        print(np.isnan(St.numpy()).any())
        print(np.isnan(Ut.numpy()).any())
        print(np.isnan(Vt.numpy()).any())

        x_soft = tf.stack([tf.math.real(x_soft), tf.math.imag(x_soft)], axis=-1)
        x_out = self.conv_t(x_soft)

        output = x_out + x_in
        output = tf.complex(output[:, :, :, :, 0], output[:, :, :, :, 1])

        return output

 

if __name__ == '__main__':
    with tf.GradientTape() as g:
        # x_in  = tf.constant(np.load('1.npy'))
        x_in  = tf.ones([1, 5, 5, 5], dtype=tf.complex64)
        g.watch(x_in)

        net = exam_net()
        x_out = net(x_in)

        grad = g.gradient(x_out, [x_in, net.trainable_variables])
        print(grad[1][0])