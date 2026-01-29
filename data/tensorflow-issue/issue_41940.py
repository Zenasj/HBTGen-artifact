# tf.random.uniform((M, M), dtype=tf.float64), tf.random.uniform((M, 1), dtype=tf.float64) ← M inferred as 2048 from issue context

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, L=None):
        """
        Implements the fusion of discussed models ModelA, ModelA1, ModelB, ModelC and also
        includes a custom (naive) triangular solve implementation 'my_triangular_solve' as a submodule.

        L: lower-triangular matrix input of shape [M,M] dtype tf.float64
        """
        super().__init__()

        self.M = None
        if L is not None:
            self.M = L.shape[-1]

        # Save the input matrix L (lower triangular)
        self.L = L

        # ModelA style precomputed inverse using triangular_solve once and cached
        if L is not None:
            self.invL_precomputed = tf.linalg.triangular_solve(L, tf.eye(self.M, dtype=tf.float64), lower=True)
        else:
            self.invL_precomputed = None

    def call(self, inputs):
        """
        Forward method that:
        - inputs: a tensor `b` of shape [M,1], dtype tf.float64
        - computes:
            * invL via triangular_solve computed inline (like ModelA1 and ModelC, uses tf.linalg.triangular_solve)
            * invL_precomputed (cached from init) used like ModelA
            * invL_inv computed via tf.linalg.inv (ModelB)
            * solution with naive custom solve 'my_triangular_solve'
        Finally compares the results of:
         matmul(invL, b), matmul(invL_precomputed, b),
         matmul(invL_inv, b), and naive triangular solve result,
        returning a dictionary containing all results and boolean checks if all close within tolerance.

        This fused approach demonstrates the different ways and the comparisons discussed in the issue.

        Assumptions:
         - inputs is a 2D tensor of shape [M,1], float64 as per issue.
         - self.L is set (otherwise raises error).
        """
        b = inputs
        if self.L is None:
            raise ValueError("Model requires lower-triangular matrix L set at init.")

        M = self.M if self.M is not None else tf.shape(self.L)[-1]

        # 1. invL computed inline (ModelA1 and ModelC style)
        # Use tf.eye with constant NumPy eye to trigger constant folding as per discussion
        eye_np = tf.constant(tf.eye(M, dtype=tf.float64).numpy(), dtype=tf.float64)
        invL_inline = tf.linalg.triangular_solve(self.L, eye_np, lower=True)

        # 2. invL_precomputed (ModelA style), cached at init (if available)
        if self.invL_precomputed is None:
            invL_precomputed = tf.linalg.triangular_solve(self.L, eye_np, lower=True)
        else:
            invL_precomputed = self.invL_precomputed

        # 3. invL_inv using explicit inverse (ModelB)
        invL_inv = tf.linalg.inv(self.L)

        # 4. Custom naive triangular solve (my_triangular_solve)
        # This is a simplified loop implementation illustrating the naive solver from chunk 5.
        def my_triangular_solve(A, b):
            S = tf.shape(A)[0]
            ret = tf.zeros(S, dtype=tf.float64)

            # Iterate over rows i, accumulate sums for previous solved elements
            for i in tf.range(S):
                acc = tf.reduce_sum(A[i,:] * ret)
                ret = tf.tensor_scatter_nd_update(ret, [[i]], (b[i, 0] - acc) / A[i,i])
            return tf.reshape(ret, (S,1))

        # Compute the naive solution vector (shape [M,1])
        sol_naive = my_triangular_solve(self.L, b)

        # Compute solutions by matmul with different inverses
        sol_inline = tf.matmul(invL_inline, b)          # Using inline triangular_solve inverse
        sol_precomputed = tf.matmul(invL_precomputed, b) # Using precomputed inverse at init
        sol_inv = tf.matmul(invL_inv, b)                # Using full inverse

        # Compare solutions pairwise within a tolerance
        atol = 1e-8
        rtol = 1e-5

        def close(a, b):
            return tf.reduce_all(tf.abs(a - b) <= atol + rtol * tf.abs(b))

        inline_vs_precomputed = close(sol_inline, sol_precomputed)
        precomputed_vs_inv = close(sol_precomputed, sol_inv)
        inv_vs_naive = close(sol_inv, sol_naive)
        all_close = inline_vs_precomputed & precomputed_vs_inv & inv_vs_naive

        # Return a dictionary with results & comparisons
        return {
            "sol_inline": sol_inline,
            "sol_precomputed": sol_precomputed,
            "sol_inv": sol_inv,
            "sol_naive": sol_naive,
            "close_inline_precomputed": inline_vs_precomputed,
            "close_precomputed_inv": precomputed_vs_inv,
            "close_inv_naive": inv_vs_naive,
            "all_close_tolerance": all_close,
        }


def my_model_function():
    # To create an instance we need a lower triangular matrix L.
    # For demo, create a random positive-definite matrix and take lower cholesky,
    # Ensuring L is lower-triangular and invertible of shape [M,M] dtype float64
    M = 2048
    tf.random.set_seed(42)
    A = tf.random.uniform((M, M), dtype=tf.float64)
    A = tf.matmul(A, A, transpose_b=True) + tf.eye(M, dtype=tf.float64)  # Make positive definite
    L = tf.linalg.cholesky(A)

    return MyModel(L)


def GetInput():
    # Returns a compatible input matching b in MyModel, i.e. shape [M,1] float64
    M = 2048
    b = tf.random.uniform((M, 1), dtype=tf.float64)
    return b

