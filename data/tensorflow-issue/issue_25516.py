# tf.random.uniform((B, H, W, C), dtype=tf.float32) <- Input shape and dtype not specified in the issue, 
# so assuming a generic 4D tensor typical for image batches

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This class implements a combined model that demonstrates the essence of the issue described:
    defining metric functions that depend on intermediate tensors computed inside the loss function.

    Since traditional Keras/TensorFlow Keras does not support metrics being dynamically created
    inside the loss function, this model encapsulates the logic in a way that mimics the original intent:
    - Computes intermediate tensors in call()
    - Stores metrics as callable functions referencing these tensors
    - Provides a method to get these metrics for compilation or evaluation.

    Note: This design is a conceptual reconstruction inspired by the original reported pattern,
    adapted for TF2.20.0 and compatible with XLA compilation.
    """

    def __init__(self):
        super().__init__()
        # Internal storage for the intermediate tensors for metrics
        self._intermediate_tensors = {}
        
        # Names of metrics, corresponding to intermediate tensors
        self.metric_names = ['m1', 'm2', 'm3']
        
        # Create metric functions (closures) that access intermediate tensors
        # These are standard callables with signature (y_true, y_pred)
        # which simply return the corresponding stored tensor.
        def make_metric_fn(name):
            # Closure that returns the stored intermediate tensor corresponding to `name`.
            def metric_fn(y_true, y_pred):
                # y_true, y_pred are inputs from Keras metric call,
                # but metric values come from stored tensors.
                return self._intermediate_tensors.get(name, tf.constant(0.0))
            metric_fn.__name__ = name
            return metric_fn
        
        self.metrics_fns = [make_metric_fn(name) for name in self.metric_names]

    def call(self, inputs, training=None):
        # This example assumes a simple regression task where input shape is arbitrary
        # and output is predicted tensor of same shape
        # For demonstration, define y_pred = inputs (identity)
        # This lets us compute metrics based on (y_true - y_pred) as in the example.
        
        # But since no y_true here, we'll assume inputs contain y_true and y_pred concatenated,
        # or just simulate y_pred for demonstration.
        # Because Keras doesn't pass y_true to call, only inputs.
        #
        # Therefore, to simulate, assume inputs is a tuple (x, y_true).
        # We'll unpack below in a helper method to make inputs compatible.
        
        # For XLA compatibility we avoid Python control flow here.

        # Instead, we'll just return input for demo.
        return inputs

    def loss(self, y_true, y_pred):
        """
        Computes the loss and simultaneously sets intermediate tensors used for metrics.
        This mimics the closure approach from the issue.
        """
        diff = y_true - y_pred
        
        # Calculate intermediate tensors
        t1 = tf.reduce_mean(diff)
        m2 = tf.reduce_max(diff)
        m3 = tf.reduce_min(diff)
        
        # Store them for metric functions to access
        self._intermediate_tensors['m1'] = t1
        self._intermediate_tensors['m2'] = m2
        self._intermediate_tensors['m3'] = m3
        
        # Return the actual loss (sum of squared differences)
        return tf.reduce_sum(tf.square(diff))
    
    def get_metrics(self):
        # Return the list of metric functions in the order of metric_names
        return self.metrics_fns


def my_model_function():
    """
    Create and return an instance of MyModel.
    """
    return MyModel()


def GetInput():
    """
    Returns a tuple (inputs, y_true) compatible with MyModel.
    Since MyModel expects separate inputs and labels,
    we provide random float tensors of shape [batch, height, width, channels].
    Assuming batch=4, H=32, W=32, C=3 as a typical image input example.
    """

    B, H, W, C = 4, 32, 32, 3

    # Create random input tensor (x)
    x = tf.random.uniform((B, H, W, C), dtype=tf.float32)

    # Create random labels y_true with same shape (for simplicity)
    y_true = tf.random.uniform((B, H, W, C), dtype=tf.float32)

    return (x, y_true)

