# tf.random.uniform((B,), dtype=tf.int32) ← Assuming input is a 1-D tensor of row_splits or similar for RaggedStructure

import tensorflow as tf

class RaggedStructure(tf.CompositeTensor):
    def __init__(self, row_splits, cached_row_lengths=None, cached_value_rowids=None, cached_nrows=None, internal=False):
        """
        A TensorFlow CompositeTensor representing the row partitioning of a RaggedTensor.
        This encapsulates the 'structure' (row splits) independent of the values.
        Args:
          row_splits: A 1-D int32 or int64 tensor specifying the row splits.
          cached_row_lengths: Optional cached row lengths tensor.
          cached_value_rowids: Optional cached value row IDs tensor.
          cached_nrows: Optional cached number of rows.
          internal: For internal use.
        """
        super().__init__()
        self._row_splits = row_splits
        self._cached_row_lengths = cached_row_lengths
        self._cached_value_rowids = cached_value_rowids
        self._cached_nrows = cached_nrows

        # Basic static checks can be added here if needed
        if row_splits is None:
            raise ValueError("row_splits cannot be None")

    @property
    def row_splits(self):
        return self._row_splits

    def row_lengths(self):
        """Compute or return cached row lengths.

        row_lengths are computed as differences of row_splits segments.
        """
        # If cached_row_lengths not present, compute and cache
        if self._cached_row_lengths is None:
            # row_lengths = row_splits[1:] - row_splits[:-1]
            self._cached_row_lengths = self._row_splits[1:] - self._row_splits[:-1]
        return self._cached_row_lengths

    def value_rowids(self):
        """Compute or return cached value row ids.

        value_rowids maps each value to its row index.
        """
        if self._cached_value_rowids is None:
            # Generate segment_ids from row_splits
            self._cached_value_rowids = tf.ragged.row_splits_to_row_ids(self._row_splits)
        return self._cached_value_rowids

    def nrows(self):
        """Compute or return cached number of rows."""
        if self._cached_nrows is None:
            self._cached_nrows = tf.shape(self._row_splits)[0] - 1
        return self._cached_nrows

    @classmethod
    def from_row_lengths(cls, row_lengths):
        """Factory method to create RaggedStructure from row lengths."""
        # row_splits = tf.concat([[0], cumsum(row_lengths)], axis=0)
        row_splits = tf.concat([[0], tf.math.cumsum(row_lengths, exclusive=False)], axis=0)
        return cls(row_splits)

    def with_row_splits(self, new_row_splits):
        """Return a new RaggedStructure with the same cached values but new row_splits."""
        return RaggedStructure(
            new_row_splits,
            cached_row_lengths=None,
            cached_value_rowids=None,
            cached_nrows=None
        )

    def __repr__(self):
        return f"RaggedStructure(row_splits={self._row_splits})"


class MyModel(tf.keras.Model):
    """
    Model encapsulating both RaggedTensor and RaggedStructure logic.
    This follows the suggestion in the issue to decouple RaggedStructure from RaggedTensor,
    representing structure independently and allowing value replacement.

    For demonstration, this model accepts a RaggedStructure input and values tensor,
    creates a RaggedTensor from them, and also maintains the RaggedStructure separately.
    It returns both the RaggedTensor and also
    confirms that its internal row_lengths match those from the RaggedStructure.

    Output is a dictionary with:
      - 'ragged_tensor': the constructed tf.RaggedTensor
      - 'row_lengths_equal': boolean tensor asserting row_lengths equality between RT and RS
    """

    def __init__(self):
        super().__init__()
        # No trainable parameters required for this conceptual model

    def call(self, inputs):
        """
        inputs: tuple of (values, ragged_row_splits)
          values: tf.Tensor with arbitrary shape `[N, ...]`,
                  typically concatenated inner values for ragged.
          ragged_row_splits: 1-D tf.Tensor of type int32/int64 indicating row splits

        Returns:
          dict with:
            - 'ragged_tensor': RaggedTensor constructed from values and row_splits
            - 'row_lengths_equal': boolean scalar tensor confirming RT row_lengths match RaggedStructure row_lengths
        """
        values, ragged_row_splits = inputs

        # Create RaggedStructure from row_splits
        ragged_structure = RaggedStructure(ragged_row_splits)

        # Construct RaggedTensor from values and row_splits
        rt = tf.RaggedTensor.from_row_splits(values, ragged_row_splits)

        # Validate row_lengths equality
        rt_row_lengths = rt.row_lengths()
        rs_row_lengths = ragged_structure.row_lengths()

        row_lengths_equal = tf.reduce_all(tf.equal(rt_row_lengths, rs_row_lengths))

        return {
            'ragged_tensor': rt,
            'row_lengths_equal': row_lengths_equal
        }

def my_model_function():
    # Return a MyModel instance
    return MyModel()

def GetInput():
    """
    Returns a tuple (values, ragged_row_splits) that can be fed to MyModel.

    We create a simple ragged structure with 3 rows, row lengths [2, 3, 1].
    So row_splits = [0, 2, 5, 6]

    values can be any tensor of shape (6,) matching total number of elements in all rows.
    """
    # Define row splits (shape [4])
    row_splits = tf.constant([0, 2, 5, 6], dtype=tf.int32)

    # Create values matching total values (length 6)
    values = tf.range(6)

    return (values, row_splits)

