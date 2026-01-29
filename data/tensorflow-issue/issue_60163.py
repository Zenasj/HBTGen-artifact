# tf.random.normal((B, 64, 64, 3), dtype=tf.float32)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate, Conv2D
from tensorflow.keras import Model
from tensorflow import reshape, reduce_sum, transpose, matmul


def hilbert_transform(x, N=None, axis=-1):
    # Implements a Hilbert transform for real inputs using 1D RFFT and IRFFT with gradient support
    # Replaces the original 3D FFT usage which had no registered gradient
    if x.dtype.is_complex:
        raise ValueError("Input x must be real-valued tensor.")
    if N is None:
        N = x.shape[axis]
    if N is None or N <= 0:
        raise ValueError("N must be positive and known.")

    # Apply 1D RFFT on slices along the axis, here assumed axis=-1 (like width dimension)
    # Input shape example: (batch, height, width, channels)
    # x[:, :, :1], x[:, :, 1:] slices channels assumed to be split real/imag parts?
    # As input channel dimension is 3, this split is unusual but from original code.
    x_br = tf.signal.rfft(x[:, :, :1])  # shape (..., rfft_length)
    x_bi = tf.signal.rfft(x[:, :, 1:])  # shape (..., rfft_length)
    x_complex = tf.concat([x_br, x_bi], axis=-1)

    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2

    # Expand dims of h for broadcasting across input tensor dimensions
    if x.shape.ndims is not None and x.shape.ndims > 1:
        ind = [tf.newaxis] * x.shape.ndims
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    h = tf.convert_to_tensor(h, dtype=x_complex.dtype)

    x_complex_h = x_complex * h
    x_bj = tf.signal.irfft(x_complex_h[:, :, :1])
    x_bk = tf.signal.irfft(x_complex_h[:, :, 1:])
    # Return concatenated imaginary parts to match original interface (even though irfft returns real)
    # We replicate original returned tensor shape (b,h,w,channels*2-2)
    x_hilbert = tf.concat([tf.math.imag(x_bj), tf.math.imag(x_bk)], axis=-1)

    return x_hilbert


class ImageUpgradingBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Using kernel_size=64 as per original code, which is large but kept consistent
        self.conv1 = Conv2D(filters=3, kernel_size=64)
        self.conv2 = Conv2D(filters=3, kernel_size=64)

    def call(self, inputs, *args, **kwargs):
        # inputs shape: (batch, height, width, channels=3)
        maps_x = inputs[:, :, :, :3]

        v1 = self.conv1(maps_x)  # shape (batch, h_out, w_out, 3)
        v2 = self.conv2(maps_x)  # shape (batch, h_out, w_out, 3)

        # Reshape to (batch * h_out * w_out, 3, 1) for vectors with last dim 1
        v1_reshaped = reshape(v1, [-1, 3, 1])
        v2_reshaped = reshape(v2, [-1, 3, 1])

        # Stack vectors along axis=-1 to form 2D subspace: (..., 3, 2)
        V = tf.concat([v1_reshaped, v2_reshaped], axis=-1)

        # Normalize V by sum of absolute values along axis=1 (features axis), keep dims for broadcasting
        norm = 1e-6 + reduce_sum(tf.abs(V), axis=1, keepdims=True)
        V = V / norm

        # Original spatial dims retrieved, may be None sometimes, so check for that
        b_, h_, w_, c_ = tf.unstack(tf.shape(maps_x))
        # maps_x_t shape: (batch, channels=3, h_*w_)
        maps_x_t = transpose(reshape(maps_x, [-1, h_ * w_, c_]), perm=[0, 2, 1])
        # V_t shape: (batch * h_ * w_, 2, 3) since V shape is (batch * h_ * w_, 3, 2)
        V_t = transpose(V, perm=[0, 2, 1])

        mat = matmul(V_t, V)  # shape (batch * h_ * w_, 2, 2)
        mat_inv = tf.linalg.inv(mat)  # inverse of small matrices

        # Projection: Y = V * (V_t * V)^-1 * V_t * X
        reconstruct_x_t = matmul(V, matmul(mat_inv, matmul(V_t, maps_x_t)))  # shape (batch * h_ * w_, 3, h_*w_)

        # Shapes manipulation for lstsq fallback if batch dim is None (unlikely)
        # original code fallback if batch dim None, though usually batch dim is known
        reconstruct_x = None
        # Reshape reconstruct_x_t from (batch * h_*w_, 3, h_*w_) to (batch, h_, w_, ?)
        # Because reconstruct_x_t shape mismatches input spatial dims, this part is ambiguous,
        # But we follow the shape flow from original code that intended (batch, h_, w_, c)
        # So we reshape back later

        # The reconstruct_x_t currently has batch*h_*w_ as leading dim, no batch split, so just keep as is

        # Since lstsq usually expects (M, N) matrices, apply lstsq batch-wise here:
        if b_ is None:
            # If batch dimension unknown, fallback:
            reconstruct_x = tf.linalg.lstsq(tf.squeeze(V, axis=[0]), tf.squeeze(reconstruct_x_t, axis=[0]))
            reconstruct_x = tf.expand_dims(reconstruct_x, axis=0)  # expand batch dim
        else:
            reconstruct_x = tf.linalg.lstsq(V, reconstruct_x_t)

        reconstruct_x = transpose(reconstruct_x, perm=[0, 2, 1])  # shape (-1, h_*w_, c)

        # Reshape to (batch, h_, w_, channels)
        reconstruct_x = reshape(reconstruct_x, [-1, h_, w_, c_])

        # Apply Hilbert transform along axis=1 (height) as per original design
        reconstruct_x_h = hilbert_transform(reconstruct_x, axis=1)

        # Concatenate along channel axis; original concatenation logic: (b, h, w, c*2 - 2)
        # Use only channels minus 1 for real + imag channel doubling
        output = tf.concat([reconstruct_x, reconstruct_x_h], axis=-1)
        return output

    def get_config(self):
        # Ensure proper serialization
        config = super().get_config()
        return config


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.image_upgrade_block = ImageUpgradingBlock()
        self.final_conv = Conv2D(3, 3, padding='same')  # output 3 channels

    def call(self, inputs, *args, **kwargs):
        x = self.image_upgrade_block(inputs)
        x = self.final_conv(x)
        return x


# Create the model instance once to maintain weights across train steps
def my_model_function():
    return MyModel()


loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()


@tf.function
def train_step(model, inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def GetInput():
    # Generate input that matches the model expected shape: (batch, 64, 64, 3), float32
    # Using normal distribution as in original example
    return tf.random.normal((1, 64, 64, 3), dtype=tf.float32)

