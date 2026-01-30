import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import time

import tensorflow as tf


class DebugModel(tf.keras.Model):
  def __init__(self):
    super(DebugModel, self).__init__()
    self.dense = tf.keras.layers.Dense(20, activation='relu', input_shape=(5,))

  def __call__(self, input, *args, **kwargs):
    x = self.dense(input)
    return x


def train_step(input):
  with tf.GradientTape() as tape:
    out = my_model(input)
    my_loss = -tf.reduce_mean(out)
  gradients = tape.gradient(my_loss, my_model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))
  return my_loss


def distributed_train():
  return strategy.experimental_run(train_step, train_iterator)

@tf.function
def distributed_train_tf_function():
  return strategy.experimental_run(train_step, train_iterator)


if __name__ == '__main__':
  dataset = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([3000, 16, 5])))
  N_steps = 1000

  # Plain computation
  my_model = DebugModel()
  optimizer = tf.keras.optimizers.Adam(1e-2)
  _s = time.time()
  i = 0
  for data in dataset:
    if i >= N_steps:
      break
    train_step(data)
    i += 1
  print('Computing for {}-steps, no strategy: Time: {} sec'.format(
      N_steps, time.time() - _s))


  # Mirrored strategy with @tf.function
  dataset = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([3000, 16, 5])))
  strategy = tf.distribute.MirroredStrategy()

  with strategy.scope():
    train_iterator = strategy.make_dataset_iterator(dataset)
    train_iterator.initialize()
    optimizer = tf.keras.optimizers.Adam(1e-2)
    my_model = DebugModel()

    _s = time.time()
    for s in range(N_steps):
      distributed_train_tf_function()
    print('Computing for {}-steps, strategy with tf-function: Time: {}'.format(
      N_steps, time.time() - _s))


  # Mirrored strategy without @tf.function decorator
  dataset = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([3000, 16, 5])))
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    train_iterator = strategy.make_dataset_iterator(dataset)
    train_iterator.initialize()
    optimizer = tf.keras.optimizers.Adam(1e-2)
    my_model = DebugModel()

    _s = time.time()
    for s in range(N_steps):
      distributed_train()
    print('Computing for {}-steps, strategy with no tf-function: Time: {}'.format(
      N_steps,
      time.time() - _s))