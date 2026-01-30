import tensorflow as tf
from tensorflow import keras

def mae(y_true, y_pred):
    return tf.keras.backend.mean(tf.math.abs(y_pred - y_true), axis=-1)

from tensorflow.keras.metrics import Metric
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

import tensorflow

class DirectionAccuracy(Metric):

    def __init__(self, name="direction_accuracy", dtype=None, **kwargs):
        super(Metric, self).__init__(name=name, dtype=dtype, **kwargs)
        self.total_count = self.add_weight("total_count", initializer=init_ops.zeros_initializer)
        self.match_count = self.add_weight("match_count", initializer=init_ops.zeros_initializer)
        self.direction_matches = self.add_weight("direction_matches", initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.direction_matches = math_ops.multiply(y_true, y_pred)
        count_result = math_ops.count_nonzero(tensorflow.greater_equal(self.direction_matches, 0.), dtype="float32")

        self.match_count.assign_add(count_result)
        self.total_count.assign_add(len(y_true))
        
        
    def result(self):
      return math_ops.div_no_nan(self.match_count, self.total_count)