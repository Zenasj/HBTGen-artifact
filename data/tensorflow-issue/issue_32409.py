import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

BATCH_SIZE = 4

class SimpleModel(keras.Model):

  def __init__(self):
    super(SimpleModel, self).__init__()
    self._layer = keras.layers.Dense(1)

  def call(self, inputs):
    learning_phase = keras.backend.learning_phase()
    inputs = tf.Print(inputs, [tf.constant(learning_phase.shape.ndims)],
                      "Looking at the ndims says this is a salar: ",
                      summarize=BATCH_SIZE)
    inputs = tf.Print(inputs, [learning_phase], "But this is not a scalar: ",
                      summarize=BATCH_SIZE)
    inputs = tf.Print(inputs,
                      [tf.reduce_any(learning_phase)],
                      "Even reduce_any does not create a scalar: ", summarize=BATCH_SIZE)
    return self._layer(inputs)


def get_dataset(batch_size=BATCH_SIZE):
  num_batches = 16
  num_features = 5
  inputs = np.random.random((num_batches, num_features))
  labels = np.random.random((num_batches))
  sample_weights = (np.random.random((num_batches)) > 0.5).astype(np.float32)
  dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, sample_weights))
  return dataset.batch(batch_size).take(1)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.ERROR)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  print("-------------")
  print(tf.__version__)
  print("-------------")

  model = SimpleModel()
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.005)
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  model.compile(optimizer, loss)
  model.fit(get_dataset(), verbose=0)