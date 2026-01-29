# tf.random.uniform((B, None, 32), dtype=tf.float64) ← Input is a batch of variable-length 2D features (float64)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # Two dense layers, one for each segment sum variant
        self.fc_unsorted = tf.keras.layers.Dense(units=1)
        self.fc_segment = tf.keras.layers.Dense(units=1)

    @tf.function
    def call(self, inputs, training=None):
        """
        inputs: tuple of (x, y, z)
          x: float64 Tensor, shape (batch_size, feature_dim), features to sum
          y: int64 Tensor, shape (batch_size,), segment ids for each example in x
          z: int32 scalar, number of segments for unsorted_segment_sum
          
        Note: batch_size is dynamic (None), feature_dim=32, dtype=float64 as per the provided code.
        """
        x, y, z = inputs

        # Using unsorted_segment_sum: shape of result is (z, feature_dim), but shape info is lost due to bug
        # This causes failure in Dense layer because last dimension is None
        # We fix it by explicitly setting the shape after unsorted_segment_sum call.
        x_unsorted = tf.math.unsorted_segment_sum(x, tf.squeeze(y), z)
        # Set static shape to (z, 32), assuming feature_dim=32 as in data
        # z is scalar tensor, shape [].
        # We use z as int if known; otherwise fallback to None for batch dimension
        if z.shape.rank == 0 and z.dtype.is_integer:
            # z is scalar int32 tensor - use .numpy() if in eager or tf.shape fallback in graph
            # In graph, use tf.shape
            try:
                num_segments = int(z.numpy())
            except Exception:
                num_segments = tf.shape(x_unsorted)[0]
            x_unsorted.set_shape([num_segments, 32])
        else:
            # Unknown segments - set partial shape
            x_unsorted.set_shape([None, 32])

        out_unsorted = self.fc_unsorted(x_unsorted)

        # Using segment_sum: shapes remain intact, should work normally
        x_segment = tf.math.segment_sum(x, tf.squeeze(y))
        # x_segment shape: (num_segments_defined_by_segment_ids, 32)
        out_segment = self.fc_segment(x_segment)

        # For demonstration, return difference in outputs to illustrate comparison/fusion of both approaches
        # This is an example of fusion of two models and their outputs
        diff = out_unsorted - out_segment
        return diff

def my_model_function():
    return MyModel()

def GetInput():
    """
    Return a valid tuple of inputs matching MyModel:
      x: float64 tensor of shape (N, 32), where N is total number of elements (variable)
      y: int64 tensor of shape (N,), segment indices for x
      z: int32 scalar tensor, number of segments
      
    We mimic the generator logic from the issue to create a batch of data.
    """
    # Randomly create offset and aggregation segments similar to original code:
    # offset = np.random.randint(1, 10, size=1024)
    # y = np.repeat(np.arange(1024), offset)
    # z = 1024
    offset = np.random.randint(1, 10, size=16)  # smaller for example
    y_np = np.repeat(np.arange(16), offset)
    z_np = np.int32(16)
    x_np = np.random.rand(np.sum(offset), 32).astype(np.float64)

    x = tf.convert_to_tensor(x_np, dtype=tf.float64)
    y = tf.convert_to_tensor(y_np, dtype=tf.int64)
    z = tf.convert_to_tensor(z_np, dtype=tf.int32)

    return (x, y, z)

