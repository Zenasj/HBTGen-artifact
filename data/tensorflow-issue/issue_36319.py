# tf.random.uniform((B, 5), dtype=tf.float32) ← Input shape inferred: batch_size x 5 features

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_inputs=5, num_outputs=3):
        super().__init__()
        # Simple Dense layer producing multiple outputs (num_outputs)
        self.dense = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs):
        # inputs shape: (batch_size, num_inputs)
        return self.dense(inputs)


class CustomAUC(tf.keras.metrics.AUC):
    """
    AUC metric for a single output in a multi-output scenario.
    Extends tf.keras.metrics.AUC but only evaluates on one output index.
    """
    def __init__(self, output_index, name=None, **kwargs):
        # Provide a distinct name per output
        if not name:
            name = f"custom_auc_{output_index}"
        super().__init__(name=name, **kwargs)
        self.output_index = output_index

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Select the relevant output slice
        y_true_slice = y_true[:, self.output_index]
        y_pred_slice = y_pred[:, self.output_index]
        # AUC expects predictions to be in [0,1] for ROC AUC - 
        # so we apply sigmoid activation here for safety.
        y_pred_slice = tf.math.sigmoid(y_pred_slice)
        return super().update_state(y_true_slice, y_pred_slice, sample_weight)


def my_model_function():
    # Return an instance of MyModel with default parameters (5 inputs, 3 outputs)
    return MyModel(num_inputs=5, num_outputs=3)


def GetInput():
    # Return a random tensor matching expected input shape
    # Using float32 for compatibility with common TF models
    # Shape: (batch_size=8, num_inputs=5)
    return tf.random.uniform((8, 5), minval=0, maxval=1, dtype=tf.float32)

