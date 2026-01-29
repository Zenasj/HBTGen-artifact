# tf.constant([2], dtype=tf.float16) ← Input example shape for top_singular_vector is (2,), using dtype=tf.half

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def frobenius_norm(self, m):
        return tf.math.reduce_sum(tf.square(m)) ** 0.5

    def top_singular_vector(self, m):
        # Handle the case where m is a tensor of rank 0 or 1 for SVD
        n = tf.cond(
            tf.equal(tf.rank(m), 0),
            lambda: tf.expand_dims(tf.expand_dims(m, 0), 0),
            lambda: m,
        )
        n = tf.cond(
            tf.equal(tf.rank(m), 1),
            lambda: tf.expand_dims(n, 0),
            lambda: n,
        )
        # Compute truncated SVD
        st, ut, vt = tf.linalg.svd(n, full_matrices=False)
        m_size = tf.shape(n)
        # Extract top singular vectors and reconstruct rank-1 approx
        ut = tf.reshape(ut[:, 0], [m_size[0], 1])
        vt = tf.reshape(vt[:, 0], [m_size[1], 1])
        st_mat = tf.matmul(ut, vt, transpose_b=True)

        # Remove introduced dims to match input shape
        st_mat = tf.cond(
            tf.equal(tf.rank(m), 0),
            lambda: tf.squeeze(tf.squeeze(st_mat, 0), 0),
            lambda: st_mat,
        )
        st_mat = tf.cond(
            tf.equal(tf.rank(m), 1),
            lambda: tf.squeeze(st_mat, 0),
            lambda: st_mat,
        )
        return st_mat

    def call(self, x):
        # For demonstration: compute top singular vector of input x
        # Accepts rank 0, 1, or 2+ tensor inputs with dtype float16/float32/float64
        return self.top_singular_vector(x)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Based on the test examples, input is 1D tensor of length 2 with dtype float16
    # Using random values to match dtype and shape expected by top_singular_vector
    return tf.random.uniform(shape=(2,), dtype=tf.float16)

