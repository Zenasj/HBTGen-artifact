import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np  
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

SEED = 88
tf.compat.v1.random.set_random_seed(SEED)

ds_train = tf.data.Dataset.range(0, 4)
ds_val = tf.data.Dataset.range(0, 4)
ds_train = ds_train.map(
    lambda x: tf.random.uniform([1], 0.2, 5.0)
)
ds_val = ds_val.map(
    lambda x: tf.random.uniform([1], 0.2, 5.0) 
)
for el in ds_train:
    print(el.numpy())
    # --> [0.44027787], [1.7892183], [2.8793733], [3.3438706]
for el in ds_val:
    print(el.numpy())
    # why the same here? --> [0.44027787], [1.7892183], [2.8793733], [3.3438706]

tf.random.uniform

seed

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
SEED = 88
tf.compat.v1.random.set_random_seed(SEED)
rand1 = tf.random.uniform([1], 0.2, 5.0, seed=SEED) #-->[2.7339704]
rand2 = tf.random.uniform([1], 0.2, 5.0, seed=SEED) #-->[1.4490409] different
# Same when only graph-level seed is set
rand1 = tf.random.uniform([1], 0.2, 5.0) #-->[2.9647074]
rand2 = tf.random.uniform([1], 0.2, 5.0) #-->[2.7952404] different

class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels

# Create a tf.datasets
ds = tf.data.Dataset.from_tensor_slices((fields, masks))

# plot to verify
for f, m in ds.map(Augment()).take(5):
    show_image(f.numpy(), m.numpy())