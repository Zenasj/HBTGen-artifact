# tf.random.uniform((B, 784), dtype=tf.float32)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Metric
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define a simple 3-layer MLP as per original example
        self.dense_1 = layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = layers.Dense(64, activation='relu', name='dense_2')
        self.predictions = layers.Dense(10, activation='softmax', name='predictions')
        # Instantiate the custom metric
        self.custom_metric = CustomMetric()

    def call(self, inputs, training=False):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        output = self.predictions(x)
        # Optionally, update custom metric here if desired.
        # To keep forward pass pure, metric update should happen separately. 
        return output

# CustomMetric adapted from the issue, but with the critical fix:
# We cannot name a variable 'weights' as it conflicts internally.
# So renamed 'weights_intermediate' instead of 'weights'.
class CustomMetric(Metric):
    def __init__(self, name='score', dtype=tf.float32):
        super(CustomMetric, self).__init__(name=name, dtype=dtype)
        self.true_positives = self.add_weight(
            name='true_positives',
            shape=[10],
            initializer='zeros',
            dtype=self.dtype)
        # Critical fix: rename variable to avoid conflict with built-in property
        self.weights_intermediate = self.add_weight(
            name='weights_intermediate',
            shape=[10],
            initializer='zeros',
            dtype=self.dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Placeholder: metric update logic can be implemented here based on y_true and y_pred.
        # For example purposes, do nothing.
        pass

    def result(self):
        # Return a scalar metric result.
        return tf.constant(0.0, dtype=self.dtype)

    def reset_states(self):
        # Reset the state variables to zero
        self.true_positives.assign(tf.zeros([10], dtype=self.dtype))
        self.weights_intermediate.assign(tf.zeros([10], dtype=self.dtype))

    def get_config(self):
        base_config = super(CustomMetric, self).get_config()
        return dict(list(base_config.items()))

def my_model_function():
    # Return an instance of MyModel with initialized weights (random by default)
    return MyModel()

def GetInput():
    # Input shape based on the original example: (batch_size, 784)
    # As batch size is dynamic, use batch size 8 for example.
    batch_size = 8
    return tf.random.uniform((batch_size, 784), dtype=tf.float32)

