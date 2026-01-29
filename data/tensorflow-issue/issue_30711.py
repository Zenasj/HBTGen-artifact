# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape inferred from the example: batch size B, feature size 10

import tensorflow as tf
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.utils import metrics_utils
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
import numpy as np
from tensorflow.python.keras.utils.generic_utils import to_list


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # A simple Dense layer to simulate model behavior from issue example
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

        # Custom Precision metric closely modeled on tf.keras.metrics.Precision
        # but fixed to avoid returning an Operation in update_state
        class CustomPrecision(Metric):
            def __init__(self,
                         thresholds=None,
                         top_k=None,
                         class_id=None,
                         name="custom_precision",
                         dtype=tf.float32):
                super(CustomPrecision, self).__init__(name=name, dtype=dtype)
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
                # Important fix from issue discussion:
                # Do NOT return the result of metrics_utils.update_confusion_matrix_variables here,
                # since it returns an Operation - which causes errors in TF 1.14+
                metrics_utils.update_confusion_matrix_variables({
                    metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                    metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives},
                    y_true,
                    y_pred,
                    thresholds=self.thresholds,
                    top_k=self.top_k,
                    class_id=self.class_id,
                    sample_weight=sample_weight)
                # Return None or nothing explicitly (in TF 2.x this is implied)
                # This avoids returning TF ops and compatibility issues.

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
                base_config = super(CustomPrecision, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))

        self.custom_precision = CustomPrecision()

    def call(self, inputs, training=None):
        # Forward pass through Dense layer
        logits = self.dense(inputs)

        # We assume inputs contain ground truth labels in some scenarios since
        # metric needs y_true and y_pred. For demonstration,
        # here we just return logits for prediction.
        # Metric update would typically happen via external tf.keras training loops.
        return logits

    def compute_metric(self, y_true, y_pred, sample_weight=None):
        # Convenience method to update custom precision metric and get result
        self.custom_precision.update_state(y_true, y_pred, sample_weight)
        return self.custom_precision.result()

def my_model_function():
    # Return an instance of MyModel (initialized layers and metric)
    return MyModel()

def GetInput():
    # Generate random input tensor matching the input layer shape (batch size 4, 10 features)
    # The model expects input shape (None, 10)
    # We'll use batch size 4 as a reasonable example
    return tf.random.uniform((4, 10), dtype=tf.float32)

