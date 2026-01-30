import math
import random
from tensorflow import keras
from tensorflow.keras import optimizers

def hilbert_transform(x, N=None, axis=-1):
    if x.dtype == tf.complex64 or x.dtype == tf.complex128:
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")
    x_br = tf.signal.rfft3d(x[:, :, :1])
    x_bi = tf.signal.rfft3d(x[:, :, 1:])
    x_complex = concatenate([x_br, x_bi], axis=-1)

    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    if x.shape.ndims > 1:
        ind = [tf.newaxis] * x.shape.ndims
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x_complex_h = x_complex * h
    x_bj = tf.signal.ifft3d(x_complex_h[:, :, :1])
    x_bk = tf.signal.ifft3d(x_complex_h[:, :, 1:])
    x_hilbert = concatenate([tf.math.imag(x_bj), tf.math.imag(x_bk)], axis=-1)
    return x_hilbert

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Sequential
from keras.layers import concatenate
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow import matmul, reshape, reduce_sum, transpose

def hilbert_transform(x, N=None, axis=-1):
    if x.dtype == tf.complex64 or x.dtype == tf.complex128:
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")
    x_br = tf.signal.rfft3d(x[:, :, :1])
    x_bi = tf.signal.rfft3d(x[:, :, 1:])
    x_complex = concatenate([x_br, x_bi], axis=-1)

    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    if x.shape.ndims > 1:
        ind = [tf.newaxis] * x.shape.ndims
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x_complex_h = x_complex * h
    x_bj = tf.signal.ifft3d(x_complex_h[:, :, :1])
    x_bk = tf.signal.ifft3d(x_complex_h[:, :, 1:])
    x_hilbert = concatenate([tf.math.imag(x_bj), tf.math.imag(x_bk)], axis=-1)
    return x_hilbert
class ImageUpgradingBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=64)
        self.conv2 = tf.keras.layers.Conv2D(filters=3, kernel_size=64)

    def call(self, inputs, *args, **kwargs):
        maps_x = inputs[:, :, :, :3]
        v1 = self.conv1(maps_x)
        v2 = self.conv2(maps_x)
        v1 = reshape(v1, [-1, 3, 1])  # basis vector with shape(-1,3,1)
        v2 = reshape(v2, [-1, 3, 1])
        V = concatenate([v1, v2], axis=-1)  # the 2-D column subspace with shape(_,3,2)
        V = V / (1e-6 + reduce_sum(abs(V), axis=1, keepdims=True))
        b_, h_, w_, c = maps_x.shape
        maps_x_t = transpose(reshape(maps_x, [-1, h_ * w_, c]), perm=[0, 2, 1])
        V_t = transpose(V, perm=[0, 2, 1])  
        mat = matmul(V_t, V)
        mat_inv = tf.linalg.inv(mat)
        # Projection --->  Y = V×(V_t×V)_(-1)×V_t×(X)
        reconstruct_x_t = matmul(V, matmul(matmul(mat_inv, V_t), maps_x_t))
        if b_ is None:  
            reconstruct_x = tf.linalg.lstsq(tf.squeeze(V, axis=[0]), tf.squeeze(reconstruct_x_t, axis=[0]))
            reconstruct_x = tf.expand_dims(reconstruct_x, axis=0)  # shape(1,c,h*w) why can‘t(None,c,h*w)
        else:
            reconstruct_x = tf.linalg.lstsq(V, reconstruct_x_t)
        reconstruct_x = transpose(reconstruct_x, perm=[0, 2, 1])  # shape(-1,h*w,c)
        reconstruct_x_h = hilbert_transform(reconstruct_x, axis=1)
        reconstruct_x = reshape(concatenate([reconstruct_x, reconstruct_x_h], axis=-1), [-1, h_, w_, c * 2 - 2])
        return reconstruct_x

    def get_config(self):
        config = super().get_config()
        return config
def RQIUNet():
    input_image = layers.Input(shape=(64, 64, 3), dtype="float32")
    x_4d = ImageUpgradingBlock()(input_image)
    result = Conv2D(3, 3, padding="same")(x_4d)
    model = models.Model(inputs=input_image, outputs=result)
    return model

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()


@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = RQIUNet()(inputs)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, RQIUNet().trainable_variables)
    optimizer.apply_gradients(zip(gradients, RQIUNet().trainable_variables))
    return loss


x = tf.random.normal((1, 64, 64, 3))
y = tf.random.normal((1, 64, 64, 3))

for i in range(10):
    loss = train_step(x, y)
    print(f"Step {i}, Loss: {loss:.4f}")

def hilbert_transform(x, N=None, axis=-1):
    ##Constructing complex number
    x_complex = tf.complex(x[:, :, :1], x[:, :, 1:])
    if x_complex.dtype != tf.complex64 and x_complex.dtype != tf.complex128:
        raise ValueError("x must be complex.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")
    x_t = transpose(x_complex, perm=[0, 2, 1])
    x_f = tf.signal.fft(x_t)
    x_f = transpose(x_f, perm=[0, 2, 1])

    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    if x.shape.ndims > 1:
        ind = [tf.newaxis] * x.shape.ndims
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x_f_h = x_f * h
    x_f = transpose(x_f_h, perm=[0, 2, 1])
    x_hilbert = tf.signal.ifft(x_f)
    x_hilbert = transpose(x_hilbert, perm=[0, 2, 1])
    x_hilbert = concatenate([tf.math.imag(x_hilbert), tf.math.real(x_hilbert)], axis=-1)
    return x_hilbert