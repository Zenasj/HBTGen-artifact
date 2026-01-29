# tf.random.uniform((1, 224, 224, 3), dtype=tf.int32) ← Input shape from EfficientNetB0 example (batch=1, H=224, W=224, C=3)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using mean and variance values from the reported EfficientNet normalization layer
        # As per the issue:
        # mean = [0.485, 0.456, 0.406]
        # variance (incorrectly set to stddev) = [0.229, 0.224, 0.225]
        # Correct variance should be square of stddev, i.e. variance = stddev**2
        # We'll implement both normalization approaches and compare:
        # 1. Current TF implementation uses sqrt(variance) (which was set incorrectly to stddev)
        # 2. Reference implementation uses stddev (correct)
        
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        # Given stddev values from ImageNet dataset
        self.stddev = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        # Incorrect variance values as used in the current buggy normalization layer (actually stddev)
        self.incorrect_variance = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        
        # Correct variance should be stddev squared
        self.correct_variance = tf.math.square(self.stddev)
        
    def call(self, inputs, training=False):
        """
        inputs: expected in [0, 255] uint8 or float32, shape (B, H, W, C)
        We assume inputs are floats or compatible with float32 ops.
        
        Returns:
          A dict with:
            - reference_norm: normalized inputs dividing by stddev (correct normalization)
            - current_tf_norm: normalized inputs dividing by sqrt(variance) (current TF layer behavior)
            - are_close: boolean tensor indicating if normalized outputs are numerically close (False indicates issue)
            - difference: absolute difference between the two normalized outputs
        """
        x = tf.cast(inputs, tf.float32)
        x = x / 255.0  # rescale to [0,1]
        
        # Reference normalization (correct): (x - mean) / stddev
        ref_norm = (x - self.mean) / self.stddev
        
        # Current TF normalization layer (incorrect variance used)
        # Layer normalizes as: (x - mean) / sqrt(variance)
        # Here 'variance' is actually set to stddev, so sqrt(variance) = sqrt(stddev), which is wrong
        current_norm = (x - self.mean) / tf.sqrt(self.incorrect_variance)
        
        # Compute boolean tensor indicating if the two normalized outputs are close within tolerance
        are_close = tf.reduce_all(tf.abs(ref_norm - current_norm) < 1e-6)
        
        # Compute absolute difference tensor for visualizing discrepancy per pixel/channel
        difference = tf.abs(ref_norm - current_norm)
        
        # Return a dict of outputs for inspection
        return {
            "reference_norm": ref_norm,
            "current_tf_norm": current_norm,
            "are_close": are_close,
            "difference": difference,
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor of shape (1, 224, 224, 3), dtype int32, values [0,255]
    # Matches the example input shape and dtype from the issue's reproducible code.
    tf.random.set_seed(42)  # for reproducibility
    return tf.random.uniform((1, 224, 224, 3), minval=0, maxval=256, dtype=tf.int32, seed=42)

