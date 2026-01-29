# tf.random.uniform((2, 2, 3), dtype=tf.float32) ← inferred input shape based on values shape and shape param

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No additional layers needed; this model mimics the logic of clip_by_norm applied on IndexedSlices.
        # We implement clipping by norm ourselves for demonstration.

    def call(self, inputs):
        """
        inputs: tuple (values, indices, shape, max_norm)
        values: Tensor with shape [N, M, K], here [2,2,3] (from example)
        indices: Tensor with shape [N], e.g. [2,6]
        shape: Tensor with rank 3 indicating total shape (e.g. [10,2,3] or huge values)
        max_norm: scalar float
        """

        values, indices, shape, max_norm = inputs

        # Create IndexedSlices
        indexed_slices = tf.IndexedSlices(values, indices, dense_shape=shape)

        # clip_by_norm logic for IndexedSlices: 
        # normal clip_by_norm aggregates norm over axes=None by default (i.e. all but batch dims)
        # Here, just clip each value vector by norm max_norm individually.

        values_f = tf.convert_to_tensor(indexed_slices.values, dtype=tf.float32)
        norms = tf.norm(values_f, axis=-1, keepdims=True)  # shape: (N,M,1)
        desired = tf.clip_by_norm(values_f, clip_norm=max_norm, axes=None)

        # Note: clip_by_norm will scale vectors with norm > max_norm to have norm max_norm

        # Because values is rank 3 and we want norm on last axis, values shape is (batch?, 2, 3)
        # Actually in example values shape is (2, 2, 3) - two slices, each corresponding to indices.

        # Replace values with clipped values:
        clipped_values = desired

        # Return as tensor converting from IndexedSlices with clipped values
        # Construct new IndexedSlices with clipped values:
        clipped_indexed_slices = tf.IndexedSlices(clipped_values, indexed_slices.indices, indexed_slices.dense_shape)

        # Convert to dense tensor
        clipped_tensor = tf.convert_to_tensor(clipped_indexed_slices)

        return clipped_tensor

def my_model_function():
    # Create an instance of the model
    return MyModel()

def GetInput():
    # We provide inputs matching the example:
    # values: [[[ -3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
    #          [[0.0, 2.0, 0.0], [0.0, 0.0, -1.0]]]
    # shape: [10, 2, 3] is used in safe case (selected arbitrarily to avoid overflow)
    # indices: [2, 6]
    # max_norm: 4.0

    values = tf.constant([[[-3.0, 0.0, 0.0],
                           [4.0, 0.0, 0.0]],
                          [[0.0, 2.0, 0.0],
                           [0.0, 0.0, -1.0]]], dtype=tf.float32)
    indices = tf.constant([2, 6], dtype=tf.int32)
    shape = tf.constant([10, 2, 3], dtype=tf.int64)  # smaller shape to avoid int64 overflow
    max_norm = tf.constant(4.0, dtype=tf.float32)

    return (values, indices, shape, max_norm)

