from tensorflow import keras

import numpy as np
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, training=True):
        # this model returns a dictionary of values
        return {
            "a": tf.constant([5]),
            "b": tf.constant([5]),
        }

    def train_step(self, data):
        # each sample also consists of a dictionary of values
        y_true = {"a": tf.constant([3]),
                  "b": tf.constant([4])}
        y_pred = self(4)

        # the metrics on this compiled metric container each care about
        # only one of the values in the dictionaries y_true/y_pred
        self.compiled_metrics.update_state(y_true, y_pred)

        return {m.name: m.result() for m in self.metrics}

class MetricForA(tf.keras.metrics.Metric):
    def __init__(self, name="metricForA", **kwargs):
        super(MetricForA, self).__init__(name=name, **kwargs)
        self.value = self.add_weight(name='value', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.value.assign_add((y_true["a"] - y_pred["a"]) ** 2)

    def result(self):
        return self.value

    def reset_states(self):
        self.value.assign(0)

class MetricForB(tf.keras.metrics.Metric):
    def __init__(self, name="metricForB", **kwargs):
        super(MetricForB, self).__init__(name=name, **kwargs)
        self.value = self.add_weight(name='value', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.value.assign_add((y_true["b"] - y_pred["b"]) ** 2)

    def result(self):
        return self.value

    def reset_states(self):
        self.value.assign(0)

if __name__ == "__main__":
    model = Model()
    metrics = [MetricForA(), MetricForB()]

    model.compile(metrics=metrics)

    # this results in an error, showing us that the dictionaries passed to update_state
    # are no longer dictionaries
    model.train_step([])