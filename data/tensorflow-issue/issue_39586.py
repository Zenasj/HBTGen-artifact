import math
import tensorflow as tf
from tensorflow import keras

# fix for miou calculation when using sparse_categorical _cross_entropy 
    # https://github.com/tensorflow/tensorflow/issues/32875

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    @tf.function
    def __call__(self, y_true, y_pred, sample_weight=None):
        self.y_pred = tf.math.argmax(self.y_pred, axis=-1)
        return super().__call__(y_true, self.y_pred, sample_weight=sample_weight)