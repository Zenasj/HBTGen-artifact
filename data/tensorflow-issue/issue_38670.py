# tf.random.uniform((B, ...), dtype=tf.float32)  ← Assumed input is a batch of predictions and labels vectors/tensors, shape (Batch, VectorDim). 
# Since metric operates on y_true, y_pred pairs, input to MyModel will be a tuple (y_true, y_pred) with matching shapes.

import tensorflow as tf

class DirectionAccuracyMetric(tf.keras.metrics.Metric):
    def __init__(self, name="direction_accuracy", dtype=tf.float32, **kwargs):
        super(DirectionAccuracyMetric, self).__init__(name=name, dtype=dtype, **kwargs)
        # Track counts as float32 variables
        self.total_count = self.add_weight(name="total_count", initializer="zeros", dtype=dtype)
        self.match_count = self.add_weight(name="match_count", initializer="zeros", dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute element-wise product of y_true and y_pred
        direction_matches = tf.multiply(y_true, y_pred)
        # Count how many entries are >= 0, i.e., direction matches or no sign flips
        matches = tf.math.greater_equal(direction_matches, 0.0)
        count_result = tf.cast(tf.math.count_nonzero(matches), self.dtype)

        # Increase counters
        self.match_count.assign_add(count_result)

        # Update total count: number of samples in batch (assuming batch major, count all elements)
        # We interpret total_count as total number of elements considered = batch_size * vector_dim
        batch_count = tf.cast(tf.size(y_true), self.dtype)
        self.total_count.assign_add(batch_count)

    def result(self):
        # Return ratio of matching directions out of total elements, safely divide
        return tf.math.divide_no_nan(self.match_count, self.total_count)

    def reset_states(self):
        # Reset all counts
        self.total_count.assign(0.0)
        self.match_count.assign(0.0)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Instantiate the DirectionAccuracy metric as a submodule
        self.direction_accuracy_metric = DirectionAccuracyMetric()

    @tf.function
    def call(self, inputs, training=None):
        """
        inputs: tuple of (y_true, y_pred), both tf.Tensor with same shape
        Returns:
           scalar tensor containing the direction accuracy computed over the input batch
        """
        y_true, y_pred = inputs
        # Update the metric state
        self.direction_accuracy_metric.update_state(y_true, y_pred)
        # Return the current computed metric result
        return self.direction_accuracy_metric.result()

def my_model_function():
    # Return a fresh instance of the model (and metric)
    return MyModel()

def GetInput():
    # Generate a random input pair (y_true, y_pred) to feed into the model.
    # Assume inputs are batch of continuous values, shape: (batch_size=32, vector_dim=10)
    batch_size = 32
    vector_dim = 10

    # y_true values can be -1 or 1 (to represent direction)
    y_true = tf.random.uniform((batch_size, vector_dim), minval=0, maxval=2, dtype=tf.int32)
    y_true = tf.cast(y_true * 2 - 1, tf.float32)  # Convert to -1 or +1

    # y_pred values as floats between -1 and +1, simulating predicted directions
    y_pred = tf.random.uniform((batch_size, vector_dim), minval=-1.0, maxval=1.0, dtype=tf.float32)

    return (y_true, y_pred)

