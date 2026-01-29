# tf.random.uniform((B, 36, 36, 6), dtype=tf.float32)  ← inferred input shape and dtype from Deepphys call signature

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize Deepphys model layers
        self.a_1 = tf.keras.layers.ZeroPadding2D(1)
        self.a_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, name='a_2')

        self.a_4 = tf.keras.layers.ZeroPadding2D(1)
        self.a_5 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, name='a_5')

        self.a_7 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)

        self.a_8 = tf.keras.layers.ZeroPadding2D(1)
        self.a_9 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, name='a_7')

        self.a_11 = tf.keras.layers.ZeroPadding2D(1)
        self.a_12 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, name='a_12')

        self.att_conv_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='att_1')
        self.att_conv_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='att_2')

        self.m_1 = tf.keras.layers.ZeroPadding2D(1)
        self.m_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid', name='m_2')

        self.m_4 = tf.keras.layers.ZeroPadding2D(1)
        self.m_5 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, name='m_5')

        # Replacing AveragePooling2D with Conv2D with stride 2 as in original (commented out) code
        self.m_7 = tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=2)

        self.m_8 = tf.keras.layers.ZeroPadding2D(1)
        self.m_9 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, name='m_9')

        self.m_11 = tf.keras.layers.ZeroPadding2D(1)
        self.m_12 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, name='m_12')

        self.m_14 = tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=2)

        self.f_1 = tf.keras.layers.Flatten()
        self.f_2 = tf.keras.layers.Dense(256)
        self.f_3 = tf.keras.layers.Dense(1)

    @tf.function(input_signature=[tf.TensorSpec([None, 36, 36, 6], tf.float32)])
    def call(self, inputs):
        # Inputs split into two parts along channel dimension
        _in = tf.split(inputs, 2, axis=3)  # each shape: (B, 36, 36, 3)

        # First branch A processing
        A = self.a_1(_in[1])
        A = self.a_2(A)
        A = tf.keras.activations.tanh(A)

        A = self.a_4(A)
        A = self.a_5(A)
        A = tf.keras.activations.tanh(A)

        M1 = self.att_conv_1(A)
        # Normalize M1 so that sum(abs(M1))*2 = 1 in shape broadcastable form
        norm = tf.math.reduce_sum(tf.abs(M1))
        norm = 2 * norm
        norm = tf.reshape(norm, [1, 1, 1, 1])
        M1 = M1 / norm

        A = self.a_7(A)
        A = self.a_8(A)
        A = self.a_9(A)
        A = tf.keras.activations.tanh(A)

        A = self.a_11(A)
        A = self.a_12(A)
        A = tf.keras.activations.tanh(A)

        M2 = self.att_conv_2(A)
        norm = tf.math.reduce_sum(tf.abs(M2))
        norm = 2 * norm
        norm = tf.reshape(norm, [1, 1, 1, 1])
        M2 = M2 / norm

        # Second branch M processing
        M = self.m_1(_in[0])
        M = self.m_2(M)
        M = tf.keras.activations.tanh(M)

        M = self.m_4(M)
        M = self.m_5(M)

        # Compute matmul-like using broadcast multiply due to shapes:
        # Since shapes are (1,36,36,36) and M1 is (B,H,W,C), replicate 1s tensor for batch axis
        ones_1 = tf.ones([tf.shape(M)[0], 36, 36, 36], dtype=M.dtype)
        # Using elementwise multiply to simulate batch matmul for each spatial location channel dimension:
        # This is a workaround since direct batch matmul over 4D tensors is ambiguous
        # Using multiply as in original code; no tf.matmul used
        mat_1 = ones_1 * M1  # Shape (B,36,36,36)
        g1 = mat_1 * M
        M = tf.keras.activations.tanh(g1)

        M = self.m_7(M)

        M = self.m_8(M)
        M = self.m_9(M)
        M = tf.keras.activations.tanh(M)

        M = self.m_11(M)
        M = self.m_12(M)

        ones_2 = tf.ones([tf.shape(M)[0], 18, 18, 18], dtype=M.dtype)
        mat_2 = ones_2 * M
        g2 = mat_2 * M
        M = tf.keras.activations.tanh(g2)

        M = tf.keras.activations.tanh(M)

        M = self.m_14(M)

        F = self.f_1(M)
        F = self.f_2(F)
        F = self.f_3(F)

        return F


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return random input tensor of shape (batch_size=1, 36,36,6) with dtype float32
    return tf.random.uniform((1, 36, 36, 6), dtype=tf.float32)

