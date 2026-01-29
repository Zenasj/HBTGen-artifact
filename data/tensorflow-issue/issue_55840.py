# tf.random.uniform((B=1, H=5, W=5, C=5), dtype=tf.complex64) ← inferred input shape from example ([1,5,5,5], complex64)
import tensorflow as tf
import numpy as np

class CONV_OP(tf.keras.layers.Layer):
    def __init__(self, n_f=32, ifactivate=False):
        super(CONV_OP, self).__init__()
        self.mylayers = []
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        if ifactivate:
            self.mylayers.append(tf.keras.layers.ReLU())
        self.seq = tf.keras.Sequential(self.mylayers)

    def call(self, input):
        res = self.seq(input)
        return res


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__(name='exam_net')
        self.conv = CONV_OP(n_f=2, ifactivate=False)
        self.conv_t = CONV_OP(n_f=2, ifactivate=False)
        # Trainable threshold coefficient (scalar) with initial value -2
        self.thres_coef = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=True, name='thres_coef')

    def call(self, x_in):
        # x_in: shape [batch, Nt, Nx, Ny], dtype complex64
        batch, Nt, Nx, Ny = x_in.shape

        # Convert complex input to two channels (real and imag) on last axis
        x_in_real_imag = tf.stack([tf.math.real(x_in), tf.math.imag(x_in)], axis=-1)  # shape [B, Nt, Nx, Ny, 2]

        # Apply first Conv3D block (operates on last dimension = 2 channels)
        x = self.conv(x_in_real_imag)  # shape [B, Nt, Nx, Ny, 2]

        # Recombine channels to complex tensor again for SVD
        x_c = tf.complex(x[..., 0], x[..., 1])  # shape [B, Nt, Nx, Ny] complex64

        # Perform SVD on the last two dims (Nx, Ny)
        # tf.linalg.svd expects a matrix or batch of matrices, we need to confirm dimensions
        # x_c shape: [B, Nt, Nx, Ny] treat as batch of B*Nt matrices size Nx by Ny
        # Reshape to [B*Nt, Nx, Ny] for SVD
        x_c_reshape = tf.reshape(x_c, [batch * Nt, Nx, Ny])
        S, U, V = tf.linalg.svd(x_c_reshape, compute_uv=True, full_matrices=True)

        # Threshold based shrinkage on singular values:
        # thres = sigmoid(thres_coef) * max singular value per matrix (S[..., 0])
        thres_scalar = tf.sigmoid(self.thres_coef)  # scalar in (0,1)
        thres = thres_scalar * S[:, 0]  # shape [B*Nt], max singular value per matrix

        # Apply threshold per singular values vector S[i]
        # S is shape [B*Nt, min(Nx, Ny)], relu(S - thres) per row
        # Expand thres to subtract per singular values vector
        thres_expanded = tf.expand_dims(thres, axis=1)  # shape [B*Nt, 1]
        S_thresholded = tf.nn.relu(S - thres_expanded)  # shrink singular values

        # Construct diagonal matrix for thresholded singular values
        St = tf.linalg.diag(S_thresholded)  # shape [B*Nt, Nx, Ny] (square diag matrix padded to Nx/Ny?)

        # Cast St to complex64 to match dimensions for matmul
        St = tf.cast(St, tf.complex64)

        # Conjugate transpose of V: V shape [B*Nt, Ny, Ny] full_matrices=True ensured square
        V_conj = tf.math.conj(tf.transpose(V, perm=[0, 2, 1]))  # shape [B*Nt, Ny, Ny]

        # Compute US = U * St
        US = tf.linalg.matmul(U, St)  # shape [B*Nt, Nx, Ny]

        # x_soft = US * V_conj
        x_soft = tf.linalg.matmul(US, V_conj)  # shape [B*Nt, Nx, Ny]

        # Reshape back to original 5D shape [B, Nt, Nx, Ny]
        x_soft = tf.reshape(x_soft, [batch, Nt, Nx, Ny])

        # Debug prints for NaN existance (commented out since no side-effects preferred in TF functions)
        # print(np.isnan(St.numpy()).any())
        # print(np.isnan(U.numpy()).any())
        # print(np.isnan(V.numpy()).any())

        # Convert x_soft complex back to two channels last axis
        x_soft_real_imag = tf.stack([tf.math.real(x_soft), tf.math.imag(x_soft)], axis=-1)  # shape [B, Nt, Nx, Ny, 2]

        # Pass through second Conv3D block
        x_out = self.conv_t(x_soft_real_imag)  # shape [B, Nt, Nx, Ny, 2]

        # Add residual connection to input converted to two channels real/imag
        output_sum = x_out + x_in_real_imag  # shape [B, Nt, Nx, Ny, 2]

        # Return complex tensor output
        output = tf.complex(output_sum[..., 0], output_sum[..., 1])  # shape [B, Nt, Nx, Ny] complex64

        return output


def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()


def GetInput():
    # Return a random input tensor suitable for MyModel call:
    # Shape inferred from example and code: [batch=1, Nt=5, Nx=5, Ny=5], dtype complex64
    # Use uniform random complex values between -1 and 1
    real_part = tf.random.uniform((1, 5, 5, 5), minval=-1.0, maxval=1.0, dtype=tf.float32)
    imag_part = tf.random.uniform((1, 5, 5, 5), minval=-1.0, maxval=1.0, dtype=tf.float32)
    return tf.complex(real_part, imag_part)

