from tensorflow import keras

import tensorflow as tf
from tensorflow.python.keras.backend import track_variable

class MyCustomGlobalMetric(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_true = tf.Variable(name='y_true', shape=tf.TensorShape(None), dtype=tf.dtypes.float32,
                                  initial_value=[0.],
                                  validate_shape=False, synchronization=tf.VariableSynchronization.ON_READ,
                                  aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.y_pred = tf.Variable(name='y_pred', shape=tf.TensorShape(None), dtype=tf.dtypes.float32,
                                  initial_value=[0.],
                                  validate_shape=False, synchronization=tf.VariableSynchronization.ON_READ,
                                  aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self._non_trainable_weights.append(self.y_true)
        self._non_trainable_weights.append(self.y_pred)
        track_variable(self.y_true)
        track_variable(self.y_pred)
        self.was_called = self.add_weight('was_called', initializer=tf.keras.initializers.zeros, dtype=tf.dtypes.uint8,
                                          synchronization=tf.VariableSynchronization.ON_READ,
                                          aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    def update_state(self, y_true, y_pred, **kwargs):
        y_true = K.flatten(K.cast(y_true, self.y_true.dtype))
        y_pred = K.flatten(K.cast(y_pred, self.y_true.dtype))
        a = self.y_true.assign(tf.concat([self.y_true, y_true], axis=0))
        b = self.y_pred.assign(tf.concat([self.y_pred, y_pred], axis=0))
        c = self.was_called.assign(1)
        return tf.group(a, b, c)

    def result(self):
        return tf.cond(self.was_called >= 1, lambda: some_custom_metric(self.y_true[1:], self.y_pred[1:]), lambda: 0.)

    def reset_states(self):
        super().reset_states()
        self.y_true.assign(tf.constant([0.]))
        self.y_pred.assign(tf.constant([0.]))
        self.was_called.assign(0)