import random
from tensorflow import keras
from tensorflow.keras import layers

# dataset is something like <BatchDataset shapes: ({input: (None, 100)}, {output: (None, 10)}), types: ({input: tf.float32}, {output: tf.float32})>

# subscale model is something like
class VanillaModel(Model):

  def __init__(self, num_units, **kwargs):
    super(VanillaModel, self).__init__(**kwargs)
    self.num_units = num_units

    # One linear projection layer.
    self.dense_proj = tf.keras.layers.Dense(num_units, activation='relu')

  def call(self, features):
    """Forward pass."""
    output = self.dense_proj(features['input'])
    return {
        'output': output
    }

import tensorflow as tf
import numpy as np

class VanillaModel(tf.keras.Model):

  def __init__(self, num_units, **kwargs):
    super(VanillaModel, self).__init__(**kwargs)
    self.num_units = num_units

    # One linear projection layer.
    self.dense_proj1 = tf.keras.layers.Dense(num_units, activation='relu')
    self.dense_proj2 = tf.keras.layers.Dense(num_units, activation='relu')

  def call(self, features):
    """Forward pass."""
    proj1_output = self.dense_proj1(features['input'])
    proj2_output = self.dense_proj2(features['input'])
    return {
        'proj1_output': proj1_output,
        'proj2_output': proj2_output
    }

input_tensor = np.random.normal(size=(50, 32)).astype(np.float32)
output_tensor1 = np.random.normal(size=(50, 16)).astype(np.float32)
output_tensor2 = np.random.normal(size=(50, 16)).astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices(({'input': input_tensor}, {'proj1_output': output_tensor1, 'proj2_output': output_tensor2}))
model = VanillaModel(16)

model.compile('adam', {'proj1_output': 'mse', 'proj2_output': 'mae'})
model.fit(dataset)

import tensorflow as tf
import tensorflow_datasets as tfds

def convert_dataset(item):
    """Puts the mnist dataset in the format Keras expects, (features, labels)."""
    image = item['image']
    label = item['label']
    image = tf.dtypes.cast(image, 'float32') / 255.
    return image, label

mnist_data = tfds.load('mnist')
mnist_train, mnist_test = mnist_data['train'], mnist_data['test']
mnist_train = mnist_train.map(convert_dataset).shuffle(1000).batch(100).repeat()
mnist_test = mnist_test.map(convert_dataset).batch(100)

model.fit(mnist_train, epochs=10, validation_data=mnist_test)

ds = tf.data.Dataset.from_tensor_slices(
  ({'my_feature_1': ..., 'my_feature_2': ...}, {'my_label_1': ..., 'my_label_2': ...}))

class VanillaModel(tf.keras.Model):

  def __init__(self, num_units, **kwargs):
    super(VanillaModel, self).__init__(**kwargs)
    self.num_units = num_units

    # One linear projection layer.
    self.dense_proj1 = tf.keras.layers.Dense(num_units, activation='relu')
    self.dense_proj2 = tf.keras.layers.Dense(num_units, activation='relu')

  def call(self, features):
    """Forward pass."""
    proj1_output = self.dense_proj1(features['input'])
    proj2_output = self.dense_proj2(features['input'])
    return {
        'proj1_output': proj1_output,
        'proj2_output': proj2_output
    }

input_tensor = np.random.normal(size=(50, 32)).astype(np.float32)
output_tensor1 = np.random.normal(size=(50, 16)).astype(np.float32)
output_tensor2 = np.random.normal(size=(50, 16)).astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices(({'input': input_tensor}, {'proj1_output': output_tensor1, 'proj2_output': output_tensor2}))
dataset = dataset.batch(10)  # This needs to be called to create batches of data.
model = VanillaModel(16)

model.compile('adam', {'proj1_output': 'mse', 'proj2_output': 'mae'})
model.fit(dataset)