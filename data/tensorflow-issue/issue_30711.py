import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.utils import metrics_utils
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
import numpy as np
from tensorflow.python.keras.utils.generic_utils import to_list


class Precision(Metric):
    """This is a 1:1 copy of the code in tensorflow.python.keras.metrics."""
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name=None,
                 dtype=None):

        super(Precision, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables({
            metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
            metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives},
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        result = math_ops.div_no_nan(
            self.true_positives,
            self.true_positives + self.false_positives)
        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super(Precision, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    x = tf.keras.Input((10,))
    y_hat = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[x], outputs=[y_hat])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(0.01),
        loss='binary_crossentropy',
        metrics=[Precision()])
    # However the builtin metric works:
    # model.compile(
    #     optimizer=tf.keras.optimizers.SGD(0.01),
    #     loss='binary_crossentropy',
    #     metrics=[tf.keras.metrics.Precision()])

    X = np.random.uniform(-1, 1, size=(100, 10)).astype(np.float32)
    y = np.random.choice([0, 1], size=(100,)).astype(np.float32)
    model.fit(X, y)