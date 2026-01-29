# tf.random.uniform((B, 3), dtype=tf.float32) ← Input shape is (batch_size, 3), feature dim is 3

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, reference_values=None, covariance_matrix=None):
        """
        reference_values: list or array of shape (N_ref, 3), reference points for Gaussian similarities
        covariance_matrix: numpy array or tensor of shape (3, 3), covariance matrix
        """
        super(MyModel, self).__init__()
        # Initialize reference values and covariance inverse matrix as tensors
        # Use float32 dtype for compatibility
        if reference_values is None:
            # Provide default dummy references if none given
            reference_values = np.array([[1., 2., 3.],
                                         [4., 5., 6.]], dtype=np.float32)
        else:
            reference_values = np.vstack(reference_values).astype(np.float32)
        if covariance_matrix is None:
            covariance_matrix = np.array([[1., 0., 0.],
                                          [0., 2., 0.],
                                          [0., 0., 3.]], dtype=np.float32)
        self._reference_values = tf.convert_to_tensor(reference_values)
        # Store covariance matrix inverse (or covariance matrix if already inverse)
        # Here covariance_matrix assumed as covariance, so compute inverse
        cov_inv = np.linalg.inv(covariance_matrix)
        self._cov_inv = tf.convert_to_tensor(cov_inv.astype(np.float32))

    def call(self, inputs):
        # inputs: Tensor of shape (batch_size, 3)

        # Compute diffs of shape (N_ref, 3) - input is broadcasted to (batch_size, 3)
        # To match the original logic, expand dims so that differences between references and each input can be computed
        #
        # Actually, in original code diffs = self._reference_values - inputs
        # Here self._reference_values is (N_ref, 3)
        # inputs is (batch_size, 3)
        # To subtract each input from each reference, need broadcasting:
        #
        # Expanded shapes:
        # references: (1, N_ref, 3)
        # inputs: (batch_size, 1, 3)
        #
        # Resulting diffs: (batch_size, N_ref, 3)
        references_exp = tf.expand_dims(self._reference_values, axis=0)  # shape (1, N_ref, 3)
        inputs_exp = tf.expand_dims(inputs, axis=1)                      # shape (batch_size, 1, 3)
        diffs = references_exp - inputs_exp                              # shape (batch_size, N_ref, 3)

        # Multiply diffs by covariance inverse matrix:
        # A = diffs * cov_inv
        # cov_inv shape is (3,3), diffs is (batch_size, N_ref, 3)
        # Use batch matmul, treat last dimension as feature
        # tf.matmul expects tensors of shape (..., M, K) and (..., K, N)
        # Let M=1 here, so expand dims for matrix multiply:
        # Reshape diffs to (batch_size, N_ref, 1, 3)
        # cov_inv shape (3,3)
        # Output shape after matmul: (batch_size, N_ref, 1, 3)
        # Squeeze back to (batch_size, N_ref, 3)
        diffs_expanded = tf.expand_dims(diffs, axis=2)  # (batch_size, N_ref, 1, 3)
        A = tf.matmul(diffs_expanded, self._cov_inv)    # (batch_size, N_ref, 1, 3)
        A = tf.squeeze(A, axis=2)                        # (batch_size, N_ref, 3)

        # Element wise multiply A and diffs
        B = tf.multiply(A, diffs)                        # (batch_size, N_ref, 3)

        # Sum along feature dimension (last axis)
        dist = tf.reduce_sum(B, axis=2)                  # (batch_size, N_ref)

        exp_arg = -0.5 * dist                            # (batch_size, N_ref)

        # According to the issue, returning tf.math.exp(exp_arg) directly leads to wrong values
        # workaround is to multiply by 1, which preserves dtype and corrects the bug in older TF versions
        # This workaround is kept here for compat/resilience.
        similarities = 1.0 * tf.math.exp(exp_arg)        # (batch_size, N_ref)

        # Compute maximum similarity for each input over all reference points
        max_similarity = tf.reduce_max(similarities, axis=1)  # (batch_size,)

        return max_similarity


def my_model_function():
    # Instantiate with default dummy data to allow calling GetInput compatibility
    # Users can modify references or covariance by replacing the model instance if necessary.
    return MyModel()


def GetInput():
    # Return random input of shape (batch_size, 3)
    # Use batch=4 as an example
    batch_size = 4
    return tf.random.uniform(shape=(batch_size, 3), minval=-10, maxval=10, dtype=tf.float32)

