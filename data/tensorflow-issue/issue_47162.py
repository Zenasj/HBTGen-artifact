# tf.random.uniform(()) ← The model call ignores input and returns dict of shape (1,) elements

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # This model returns fixed outputs "a" and "b" as scalar tensors inside a dict

    def call(self, x, training=True):
        return {
            "a": tf.constant([5], dtype=tf.float32),
            "b": tf.constant([5], dtype=tf.float32),
        }

    def train_step(self, data):
        # Ground truth dictionaries:
        y_true = {
            "a": tf.constant([3], dtype=tf.float32),
            "b": tf.constant([4], dtype=tf.float32)
        }
        # Predicted dictionaries from model call
        y_pred = self(4)

        # Update the compiled metrics with these dicts
        # Notably, as reported, TF autograph converts dicts to lists internally and causes errors when metrics expect dicts.
        # We keep this as is to reflect the original intended usage.
        self.compiled_metrics.update_state(y_true, y_pred)

        # Return a dict of metric results
        return {m.name: m.result() for m in self.metrics}

class MetricForA(tf.keras.metrics.Metric):
    def __init__(self, name="metricForA", **kwargs):
        super(MetricForA, self).__init__(name=name, **kwargs)
        self.value = self.add_weight(name='value', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Expecting dicts, access key "a"
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
        # Expecting dicts, access key "b"
        self.value.assign_add((y_true["b"] - y_pred["b"]) ** 2)

    def result(self):
        return self.value

    def reset_states(self):
        self.value.assign(0)

def my_model_function():
    # Create an instance of MyModel and compile with the custom metrics MetricForA and MetricForB
    model = MyModel()
    metrics = [MetricForA(), MetricForB()]
    model.compile(metrics=metrics)
    return model

def GetInput():
    # The MyModel's call ignores input and returns fixed dict outputs,
    # so input shape is irrelevant.
    # We return a dummy tensor to fulfill input API.
    return tf.random.uniform((), dtype=tf.float32)

