import numpy as np
import tensorflow as tf

ds = (
    tf.data.Dataset.from_tensor_slices((np.arange(2))))

iterator = ds
for elem in iterator:
    print(elem)
for elem in iterator:
    print(elem)

class NumpyIterable(object):
  def __init__(self, dataset):
    self.dataset = dataset

  def __iter__(self):
    return self.dataset.as_numpy_iterator()

numpy_iterable = NumpyIterable(ds)
print("\nIterate numpy iterable first time:")
for elem in numpy_iterable:
  print(elem)
print("\nIterate numpy iterable second time:")
for elem in numpy_iterable:
  print(elem)