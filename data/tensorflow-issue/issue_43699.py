# tf.random.uniform((B, 5), dtype=tf.float32) ← Inferred input shape based on build_dataset (1024, 5)

import tensorflow as tf


class MyModel(tf.keras.Model):
    """Model with `n_outputs` outputs as a dict of outputs."""
    def __init__(self, n_outputs):
        super().__init__()
        self.hidden = tf.keras.layers.Dense(16)
        # Dictionary of output layers by name
        self.out = {f"out{n}": tf.keras.layers.Dense(1) for n in range(n_outputs)}

    def call(self, inputs):
        hidden = self.hidden(inputs)
        return {out_name: out_layer(hidden) for out_name, out_layer in self.out.items()}


class DumbMetric(tf.keras.metrics.Metric):
    """
    Custom metric that takes a param at initialization and always returns it as the result.
    Fixes issue with multiple outputs by implementing get_config to enable proper copying.
    """

    def __init__(self, name="Dumb", param=0., **kwargs):
        super().__init__(name=name, **kwargs)
        self.param = float(param)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # No-op for this dummy metric
        pass

    def result(self):
        return self.param

    def reset_states(self):
        # No state to reset, but override to be explicit
        pass

    def get_config(self):
        # Ensure proper copying of metric objects with custom param for multiple outputs
        base_config = super().get_config()
        base_config.update({"param": self.param})
        return base_config


def my_model_function(n_outputs=2):
    """
    Return an instance of MyModel with specified number of outputs.
    This matches the example's use with n_outputs=1 or 2.
    """
    return MyModel(n_outputs)


def GetInput():
    """
    Return a random tensor input that matches the expected input shape of MyModel.
    Based on snippet: input shape is (batch=16, features=5)
    We use batch 16 here as typical batch size from example.
    """
    # Use tf.random.uniform as per instruction comment style
    return tf.random.uniform((16, 5), dtype=tf.float32)

