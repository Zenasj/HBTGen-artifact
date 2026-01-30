import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf


def build_dataset(n_outputs):
    """
    Build a random dataset.
    Target is a dictionnary of `n_output` elements with keys as ["out0", "out1", ...]
    """
    input_data = np.random.normal(size=(1024, 5)).astype("float32")
    target_data = {
        f"out{n}": np.random.normal(size=(1024,)).astype("float32")
        for n in range(n_outputs)
    }
    input_dataset = tf.data.Dataset.from_tensor_slices(input_data)
    target_dataset = tf.data.Dataset.from_tensor_slices(target_data)
    dataset = tf.data.Dataset.zip((input_dataset, target_dataset)).batch(16)
    return dataset


class Model(tf.keras.Model):
    """Model with `n_outputs` outputs"""
    def __init__(self, n_outputs):
        tf.keras.Model.__init__(self)
        self.hidden = tf.keras.layers.Dense(16)
        self.out = {f"out{n}": tf.keras.layers.Dense(1) for n in range(n_outputs)}

    def call(self, inputs):
        hidden = self.hidden(inputs)
        return {out_name: out_layer(hidden) for out_name, out_layer in self.out.items()}


class DumbMetric(tf.keras.metrics.Metric):
    """This metrics takes a param at __init__ time and then always return param as result"""
    def __init__(self, name="Dumb", param=0., **kwargs):
        tf.keras.metrics.Metric.__init__(self, name=name, **kwargs)
        self.param = float(param)

    def update_state(self, y_true, y_pred, sample_weight):
        pass

    def result(self):
        return self.param


for n in (1, 2):
    # For 1 output, metrics behave as expected:
    #   - metric1 returns 1
    #   - metric2 returns 2
    #
    # For 2 outputs, metrics don't behave as expected:
    #   - metric1 returns 0 (default `param` value)
    #   - metric2 returns 0 (default `param` value)

    model = Model(n)
    model.compile(
        metrics=[
            DumbMetric(name="metric1", param=1.),
            DumbMetric(name="metric2", param=2.)
        ]
    )
    metrics = model.evaluate(build_dataset(n), return_dict=True, verbose=0)
    print(f"For {n} output(s): ", metrics)

def get_config(self):
        return {
            **tf.keras.metrics.Metric.get_config(self),
            "param": self.param
        }