import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import GRU, CuDNNGRU, Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

batch_size = 256
time_steps = 750

# datasets with almost no time delay
np_input = np.random.random((batch_size, time_steps, 512))
np_output = np.random.randint(0, 32, size=(batch_size, time_steps, 8))
def generate():
  while True:
    yield np_input, np_output
output_shapes = ((batch_size, time_steps, 512), (batch_size, time_steps, 8))


# split the output to 8 parts, calculate loss with 8 labels
def loss(y_true, y_pred):
  y_true = tf.split(y_true, num_or_size_splits=8, axis=-1)
  y_pred = tf.split(y_pred, num_or_size_splits=8, axis=-1)
  loss_func = lambda true, pred: tf.keras.losses.sparse_categorical_crossentropy(
      true, pred, from_logits=True)
  return tf.reduce_mean(
      tf.add_n([loss_func(y_true[i], y_pred[i]) for i in range(8)]))


# two v100 gpus, each of which has 16GB of memory
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
  # get dataset
  dataset = tf.data.Dataset.from_generator(generate,
                                           output_types=(tf.float32, tf.int32),
                                           output_shapes=output_shapes)
  dataset = dataset.prefetch(2)

  # create model
  inputs = tf.keras.Input(shape=(None, 512))
  rnn_outputs, rnn_state = CuDNNGRU(384,
                                    return_sequences=True,
                                    return_state=True)(inputs)
  h_1, h_2 = tf.split(rnn_outputs, num_or_size_splits=2, axis=-1)
  logits_1 = Dense(128)(Dense(256, activation='relu')(h_1))
  logits_2 = Dense(128)(Dense(256, activation='relu')(h_2))
  outputs = K.concatenate([logits_1, logits_2], axis=-1)
  model = tf.keras.Model(inputs, outputs)

  tensorboard = TensorBoard(log_dir='/workspace/exp/tf_log',
                            update_freq=500,
                            profile_batch=0)
  optimizer = Adam(lr=1e-4, clipnorm=1.)
  model.compile(optimizer=optimizer, loss=loss)

# train
model.fit(dataset, epochs=4, steps_per_epoch=1000, callbacks=[tensorboard])

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import GRU, CuDNNGRU, Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

batch_size = 384
time_steps = 750

# datasets with almost no time delay
np_input = np.random.random((batch_size, time_steps, 512))
np_output = np.random.randint(0, 32, size=(batch_size, time_steps, 1))


def generate():
  while True:
    yield np_input, np_output


output_shapes = ((batch_size, time_steps, 512), (batch_size, time_steps, 1))


# split the output to 8 parts, calculate loss with 8 labels
def loss(y_true, y_pred):
  loss_func = lambda true, pred: tf.keras.losses.sparse_categorical_crossentropy(
      true, pred, from_logits=True)
  return loss_func(y_true, y_pred)


# get dataset
dataset = tf.data.Dataset.from_generator(generate,
                                         output_types=(tf.float32, tf.int32),
                                         output_shapes=output_shapes)
dataset = dataset.prefetch(3)

# create model
inputs = tf.keras.Input(shape=(None, 512))
rnn_outputs, rnn_state = CuDNNGRU(384, return_sequences=True,
                                  return_state=True)(inputs)
logits_1 = Dense(128)(Dense(256, activation='relu')(rnn_outputs))
model = tf.keras.Model(inputs, logits_1)

optimizer = Adam(lr=1e-4, clipnorm=1.)
model.compile(optimizer=optimizer, loss=loss)

# train
model.fit(dataset, epochs=4, steps_per_epoch=1000)