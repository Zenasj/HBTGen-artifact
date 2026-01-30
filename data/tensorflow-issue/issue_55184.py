import tensorflow as tf

from typing import NamedTuple, Optional

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow_datasets as tfds


class DatasetOutputs(NamedTuple):
  image: np.array
  label: np.array
  metadata: Optional[np.array]


def _map_fn(data):
  return DatasetOutputs(
      image=tf.cast(data['image'], tf.float32) / 255.,
      label=data['label'],
      metadata=None)  # Problematic.


ds = tfds.load('mnist', split='train')
ds = ds.map(_map_fn)
data = next(iter(tfds.as_numpy(ds)))