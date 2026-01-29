# tf.random.uniform((B, ...), dtype=tf.float32) ← Input shape is assumed as (batch_size, num_classes) with float predictions in [0,1]

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import to_list


class MyModel(tf.keras.Model):
    """
    This model encapsulates the F1Score metric as a submodule and acts as a wrapper.
    Input: prediction tensor of shape (batch_size, num_classes) with floats in [0,1].
    Output: the computed F1 score scalar (or vector if multiple thresholds).
    
    This is a reconstructed F1Score Metric based on the issue discussion and code fragments.
    """

    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name="f1_score_model",
                 dtype=tf.float32):
        super(MyModel, self).__init__(name=name, dtype=dtype)
        # Initialize the inner metric object:
        # If thresholds is None, default threshold is 0.5 unless top_k is set.
        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)

        # Weights to accumulate confusion matrix stats for each threshold
        num_thresholds = len(self.thresholds)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(num_thresholds,),
            initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(num_thresholds,),
            initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(num_thresholds,),
            initializer=init_ops.zeros_initializer)

        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

    def call(self, y_pred, y_true=None, sample_weight=None, training=None):
        """
        y_pred: Tensor with shape (batch_size, num_classes), prediction probs in [0,1]
        y_true: Tensor with matching shape, ground truth labels (typically one-hot or multi-label binary)

        We define the call so that model(y_pred, y_true) returns F1 score(s).

        As tensorflow.keras.Model calls call with just one tensor input normally,
        here we'll accept a tuple input: (y_true, y_pred). For embedded use,
        y_true may be optional. We'll allow only tuple input (y_true, y_pred).
        """
        # Support tuple input (y_true, y_pred)
        # If only y_pred is passed, cannot compute F1, raise error
        if y_true is None:
            if isinstance(y_pred, (tuple, list)) and len(y_pred) == 2:
                y_true, y_pred = y_pred
            else:
                raise ValueError("Input must be (y_true, y_pred) tuple to compute F1 score.")

        # Update confusion matrix statistics weights in this forward pass
        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

        # Calculate precision and recall from accumulated weights
        precision = math_ops.div_no_nan(self.true_positives,
                                        self.true_positives + self.false_positives)
        recall = math_ops.div_no_nan(self.true_positives,
                                     self.true_positives + self.false_negatives)
        # F1 score formula: 2 * (precision * recall) / (precision + recall)
        numerator = math_ops.multiply(precision, recall)
        denominator = math_ops.add(precision, recall)
        f1 = 2 * math_ops.div_no_nan(numerator, denominator)

        # If single threshold, return scalar, else vector
        if len(self.thresholds) == 1:
            return f1[0]
        else:
            return f1

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        # Reset the variables to zero
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in
             [self.true_positives, self.false_positives, self.false_negatives]]
        )

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super(MyModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def my_model_function():
    # Return an instance of MyModel with default parameters (threshold=0.5)
    return MyModel()


def GetInput():
    # Generate a random binary ground truth tensor and random prediction probs tensor
    # Both shapes are (batch_size, num_classes)
    batch_size = 4
    num_classes = 3
    # y_true is binary labels: 0 or 1, shape (batch_size, num_classes)
    y_true = tf.random.uniform(shape=(batch_size, num_classes), minval=0, maxval=2, dtype=tf.int32)
    y_true = tf.cast(y_true, tf.float32)
    # y_pred is float probabilities in [0,1], same shape
    y_pred = tf.random.uniform(shape=(batch_size, num_classes), minval=0, maxval=1, dtype=tf.float32)
    # Return tuple (y_true, y_pred) as expected input for MyModel call
    return (y_true, y_pred)

