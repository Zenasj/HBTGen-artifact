import math
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np


class GaussianSimilaritiesLayer(tf.keras.layers.Layer):
    def __init__(self, reference_values, covariance_matrix):
        super(GaussianSimilaritiesLayer, self).__init__()
        self._reference_values = tf.convert_to_tensor(np.vstack(reference_values).astype(np.float32))
        self._cov_inv = tf.convert_to_tensor(covariance_matrix.astype(np.float32))

    def call(self, inputs):
        diffs = self._reference_values - inputs
        A = tf.matmul(diffs, self._cov_inv)
        B = tf.multiply(A, diffs)
        dist = tf.reduce_sum(B, axis=1)
        exp_arg = -0.5 * dist
        # return 1 * tf.math.exp(exp_arg)  # call() returns desired value
        return tf.math.exp(exp_arg)  # call() returns wrong value


class Potential:
    def __init__(self, session, demonstrations, covariance_matrix):
        self._in = tf.keras.layers.Input(shape=(3,))
        similarities = GaussianSimilaritiesLayer(demonstrations,
                                                 covariance_matrix)(self._in)
        max_similarity = tf.keras.layers.Lambda(tf.reduce_max)(similarities)

        self._model = tf.keras.Model(inputs=[self._in],
                                     outputs=[max_similarity])
        self._session = session

    def __call__(self, s):
        return self._model.output.eval(session=self._session, feed_dict={
            self._in: s
        })


if __name__ == '__main__':

    with tf.Session() as sess:
        sa_demonstrations = [np.array([1, 2, 3], dtype=np.float32),
                             np.array([4, 5, 6], dtype=np.float32)]
        covariance_matrix = np.array([[1, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 3]], dtype=np.float32)
        phi = Potential(sess, sa_demonstrations, covariance_matrix)
        sample_s = np.array([1, 2, 2.7], dtype=np.float32)
        print(phi([sample_s]))