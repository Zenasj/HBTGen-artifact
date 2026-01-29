# tf.constant([3]) shape inferred as (1,)
# "x" input tensor shape is (1,), dtype inferred as tf.int32 or tf.int64 from example (tf.constant([3]))
# "y" segment_ids shape is (1,), dtype tf.int64
# "z" num_segments scalar, dtype tf.int64

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters; this model wraps tf.raw_ops.UnsortedSegmentProd with added safety checks.

    def call(self, inputs):
        """
        Expects inputs as a dict:
          - x: data tensor, shape (N,), any numeric dtype
          - y: segment_ids tensor, shape (N,), dtype tf.int32 or tf.int64
          - z: num_segments scalar tensor, dtype tf.int32 or tf.int64
        
        Returns:
          tf.Tensor: output of UnsortedSegmentProd with safety checks on num_segments.
        
        This model adds a safety check to avoid i32 overflow crash when num_segments is too large.
        If num_segments > max int32, throws a Python exception instead of crashing at C++ layer.
        """
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']

        # Defensive check on z (num_segments) to avoid integer overflow in backend.
        max_int32 = tf.constant(2**31 - 1, dtype=tf.int64)
        z_int64 = tf.cast(z, tf.int64)

        # If z > max_int32, raise an error to prevent crash.
        def error_fn():
            # Raise a Python ValueError via tf.py_function
            def raise_err():
                raise ValueError(f"num_segments (z)={z_int64.numpy()} exceeds int32 max limit {max_int32.numpy()}, which can cause crash in UnsortedSegmentProd.")
            # tf.py_function runs eagerly, so we wrap above in py_function to raise error at runtime.
            return tf.py_function(raise_err, [], [])

        def compute_prod():
            # Call the original ops with casted num_segments to int32, if possible.
            # Note: tf.raw_ops.UnsortedSegmentProd expects num_segments as int32 or int64 independently.
            # We preserve dtype of z, assuming backend requires i32 compatibility.
            # Segment ids must be less than num_segments to avoid undefined behavior.
            return tf.raw_ops.UnsortedSegmentProd(
                data=x, segment_ids=y, num_segments=z)

        # Conditionally execute error or prod based on z value
        result = tf.cond(
            tf.greater(z_int64, max_int32),
            true_fn=error_fn,
            false_fn=compute_prod)
        return result

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a dict input compatible with MyModel.call
    # Let's create a simple example avoiding overflow:
    # x: tensor of shape (1,), numeric dtype (int32)
    # y: segment_ids of shape (1,), int64
    # z: num_segments = 2 (within safe range)
    x = tf.constant([3], dtype=tf.int32)
    y = tf.constant([1], dtype=tf.int64)
    z = tf.constant(2, dtype=tf.int64)
    return {'x': x, 'y': y, 'z': z}

