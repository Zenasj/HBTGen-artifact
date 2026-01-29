# tf.random.uniform((2, 3), dtype=tf.float32) ← Input shape inferred from example labels and predictions

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using tf.keras.metrics.CategoricalAccuracy with float dtype since complex dtype is unsupported.
        # The original issue involved crashing with dtype=tf.complex64 because comparison semantics for complex types are undefined.
        # Here we enforce float32 dtype for safe metric calculation.
        self.categorical_accuracy = tf.keras.metrics.CategoricalAccuracy(dtype=tf.float32)

    def call(self, y_true, y_pred, sample_weight=None):
        """
        Compute categorical accuracy metric.

        Args:
          y_true: Tensor of true labels, shape (batch_size, num_classes), int or float.
          y_pred: Tensor of predicted probabilities/logits, same shape as y_true, float32.
          sample_weight: Optional tensor for weighted accuracy.

        Returns:
          Tensor scalar metric of categorical accuracy.
        """
        # Reset states to avoid accumulation if this model is reused repeatedly
        self.categorical_accuracy.reset_states()

        # Calculate categorical accuracy using float dtype metric
        # Note: We do NOT support complex input dtypes because comparison is undefined and crashes with XLA on CUDA.
        # This is the key issue described in the GitHub issue.

        # Ensure inputs are float32 tensors
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        # sample_weight can be None or a tensor
        if sample_weight is not None:
            sample_weight = tf.convert_to_tensor(sample_weight, dtype=tf.float32)

        self.categorical_accuracy.update_state(y_true, y_pred, sample_weight)
        return self.categorical_accuracy.result()

def my_model_function():
    return MyModel()

def GetInput():
    # From the issue example, inputs are two tensors shaped (2, 3) for labels and predictions
    # and a (2, 1) sample_weight tensor.
    # We'll generate random valid inputs that fit these shapes:
    # - y_true: one-hot labels (0/1) with shape (2, 3)
    # - y_pred: predicted probabilities (floats summing roughly to 1) with shape (2, 3)
    # - sample_weight: positive floats shape (2, 1)
    
    # Generate one-hot labels for 2 samples, 3 classes
    y_true = tf.constant([[0., 0., 1.],
                          [0., 1., 0.]], dtype=tf.float32)
    
    # Generate predictions as float probabilities per class (not complex) summing roughly to 1
    y_pred = tf.constant([[0.1, 0.1, 0.8],
                          [0.05, 0.0, 0.95]], dtype=tf.float32)
    
    # Sample weights as floats
    sample_weight = tf.constant([[0.5],
                                 [0.2]], dtype=tf.float32)
    
    return (y_true, y_pred, sample_weight)

