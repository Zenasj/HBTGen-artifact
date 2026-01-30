import tensorflow as tf

x = tf.RaggedTensor.from_row_splits(tf.range(5), [0, 2, 5])
y = 2 * x
print(x.row_splits is y.row_splits)              # True
print(x.row_lengths() is y.row_lengths())  # False

class RaggedTensor(CompositeTensor):
  # changed constructor, publicly usable.
  def __init__(self, values, ragged_structure):
    self._values = values
    self._ragged_structure = ragged_structure
    # error checks
    ...

  # added property
  @property
  def ragged_structure(self):
    return self._ragged_structure 
  
  # redirect row_lengths, row_starts, value_rowids etc. to structure
  def row_lengths(self):
    return self.ragged_structure.row_lengths()

  def with_values(self, values):
    return RaggedTensor(values, self.ragged_structure)

# new class
class RaggedStructure(CompositeTensor):
  def __init__(self, row_splits, cached_row_lengths=None, cached_value_rowids=None, cached_nrows=None, internal=False):
    # most of current tf.RaggedTensor.__init__
    ...

  @classmethod
  def from_row_lengths(cls, row_lengths):
    # most of current tf.RaggedTensor.from_row_lengths
    ...

  # cache here
  def row_lengths(self, row_lengths):
    if self._cached_row_lengths is None:
      ...
    return self._cached_row_lengths