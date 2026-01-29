# tf.random.uniform((B, 1), dtype=tf.float32) ← Inferred input shape: batch size unknown, single float feature input

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model mimics a simplified version of a custom categorical feature column
    that provides a SparseTensor of IDs and a SparseTensor of weights, similar to
    the _DebuggingCategoricalColumn in the original issue.

    It takes as input a dense float tensor of shape [batch_size, 1] and outputs:
    - A dense ID tensor (bucketized categories)
    - A dense weights tensor corresponding to the IDs.

    This example illustrates processing of inputs to categorical IDs and weights,
    keeping in mind the original problem related to SparseTensor weights.

    Assumptions and reconstruction notes:
    - Input is a batch of scalar floats.
    - Bucketing boundaries are fixed as [0, 2, 4] for demonstration.
    - IDs are bucket indices for the input values.
    - Weights are fixed to 1.0 for each value, as in the original example.
    - Return dense tensors (to avoid sparse IndexedSlices in grad warnings).
    """

    def __init__(self):
        super().__init__()
        # Boundaries for bucketizing
        self.boundaries = tf.constant([0, 2, 4], dtype=tf.float32)

    def call(self, inputs):
        # inputs: [batch_size, 1], float32
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

        batch_size = tf.shape(inputs)[0]

        # Squeeze to shape [batch_size]
        squeezed = tf.squeeze(inputs, axis=-1)

        # Bucketize inputs based on boundaries, result is int32 tensor [batch_size]
        buckets = tf.bucketize(squeezed, boundaries=self.boundaries)

        # Cast bucket indices to int64 for consistency with original code
        buckets = tf.cast(buckets, tf.int64)

        # Create indices for SparseTensor-style representation: each row corresponds to batch index
        indices = tf.stack([tf.range(batch_size, dtype=tf.int64), tf.zeros(batch_size, dtype=tf.int64)], axis=1)

        # Create SparseTensor for IDs
        id_sparse = tf.sparse.SparseTensor(
            indices=indices,
            values=buckets,
            dense_shape=[tf.cast(batch_size, tf.int64), 1]
        )

        # Create SparseTensor for weights (all ones)
        weights_sparse = tf.sparse.SparseTensor(
            indices=indices,
            values=tf.ones(batch_size, dtype=tf.float32),
            dense_shape=[tf.cast(batch_size, tf.int64), 1]
        )

        # Convert sparse tensors to dense tensors to avoid IndexedSlices issues in gradients (key fix)
        id_dense = tf.sparse.to_dense(id_sparse, default_value=-1)
        weights_dense = tf.sparse.to_dense(weights_sparse, default_value=0.0)

        # Return a dictionary for clarity (could also be tuple)
        return {'ids': id_dense, 'weights': weights_dense}

def my_model_function():
    # Returns an instance of MyModel with no special initialization required
    return MyModel()

def GetInput():
    # Create a random batch of float inputs with shape [batch_size, 1]
    # Assume batch size = 4 for demonstration to mirror example in the issue
    batch_size = 4

    # Values chosen around the bucket boundaries to check bucketization
    input_values = tf.constant([[-1.0], [1.0], [3.0], [5.0]], dtype=tf.float32)

    # This can be replaced with random inputs if desired:
    # input_values = tf.random.uniform((batch_size, 1), minval=-2, maxval=6, dtype=tf.float32)

    return input_values

