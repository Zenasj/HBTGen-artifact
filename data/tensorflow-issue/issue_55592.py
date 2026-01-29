# tf.random.uniform((B, ...), dtype=...) 
# Note: Input shape and dtype are inferred as the metric is designed for scalar accumulators and counts, 
# so input will be a batch of scalars (e.g., float32 tensors).

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A keras model mimicking a metric computation that works properly with mixed precision and
    distributed strategies by using TF ops instead of Python arithmetic in result().
    
    This example is inspired by the provided Perplexity metric workaround and
    the discussion about avoiding python ops on AutoCastVariables.
    """
    def __init__(self):
        super().__init__()
        # Example accumulators to simulate tracked metric state
        self._true_sum = self.add_weight(
            name="true_sum", shape=(), initializer="zeros", dtype=tf.float32
        )
        self._count = self.add_weight(
            name="count", shape=(), initializer="zeros", dtype=tf.float32
        )
    
    def call(self, inputs, training=None):
        """
        Accepts a batch of scalar values and updates the accumulators
        to mimic metric update.
        
        Args:
            inputs: tf.Tensor of shape (B,) or (B,1) with dtype float32
        
        Returns:
            Result of the metric computation - here a scalar tensor.
        """
        # Flatten in case inputs have extra dims
        inputs = tf.reshape(inputs, [-1])
        
        # Update accumulators like a metric would do in its update_state:
        # sum of inputs and count of inputs
        batch_sum = tf.reduce_sum(inputs)
        batch_count = tf.cast(tf.size(inputs), tf.float32)
        
        self._true_sum.assign_add(batch_sum)
        self._count.assign_add(batch_count)
        
        # Return current mean as proxy output
        return self.result()
    
    def result(self):
        # Use TF math ops to avoid issues with mixed precision variables that override python ops
        # Safely handle zero count with divide_no_nan
        return tf.math.divide_no_nan(self._true_sum, self._count)
    
    def reset_states(self):
        # Resets the accumulators - typical for metric reset
        self._true_sum.assign(0.0)
        self._count.assign(0.0)

def my_model_function():
    # Return an instance of MyModel with fresh weights
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input of MyModel
    # Infer shape: batch of scalar values, let's say batch size 32, dtype float32 for metric input
    B = 32
    return tf.random.uniform((B,), dtype=tf.float32)

