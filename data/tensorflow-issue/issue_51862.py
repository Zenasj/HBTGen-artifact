import tensorflow as tf
from tensorflow import keras
import math

class PSNR(keras.metrics.Metric):
    """
    Peak Signal to Noise Ratio metric
    """
    def __init__(self, name='PSNR', **kwargs):
        super().__init__(name, **kwargs)
        self.psnr = self.add_weight(name='PSNR', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred, y_true = tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)
        mse = tf.reduce_mean(keras.metrics.mean_squared_error(y_true, y_pred))
        psnr = 10.0 * tf.divide(tf.math.log(tf.divide(10000**2, mse)), math.log(10))

        self.psnr.assign_add(psnr)

    def result(self):
        return self.psnr