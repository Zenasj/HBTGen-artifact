from tensorflow import keras

# model.py
import tensorflow as tf
from tensorflow.keras import layers


class Model(tf.keras.Model):
  def __init__(self, vocab_size, emb_size, rnn_size):
    super(Model, self).__init__()
    self.embedding = layers.Embedding(vocab_size, emb_size)
    self.rnn = DynamicRNN(rnn_size)

  def call(self, x):
    x = self.embedding(x)
    x, _ = self.rnn(x)
    return x

class DynamicRNN(layers.Layer):
  def __init__(self, rnn_size):
    super(DynamicRNNV1, self).__init__()
    self.rnn_size = rnn_size
    self.cell = layers.GRU(rnn_size, return_state=True)

  @tf.function
  def call(self, input_data):
    outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    state = tf.zeros((input_data.shape[0], self.rnn_size), dtype=tf.float32)
    for i in tf.range(input_data.shape[1]):
      print(input_data)
      output, state = self.cell(tf.expand_dims(input_data[:, i, :], 1), state)
      outputs = outputs.write(i, output)
    return tf.transpose(outputs.stack(), [1, 0, 2]), state

class DynamicRNNV2(layers.Layer):
  def __init__(self, rnn_size):
    super(DynamicRNNV2, self).__init__()
    self.rnn_size = rnn_size
    self.cell = layers.GRU(rnn_size, return_state=True)

  def call(self, input_data):
    state = tf.zeros((input_data.shape[0], self.rnn_size), dtype=tf.float32)
    outputs = []
    for i in range(input_data.shape[1]):
      output, state = self.cell(tf.expand_dims(input_data[:, i, :], 1), state)
      outputs.append(tf.expand_dims(output, 1))
    return tf.concat(outputs, axis=1), state

# train.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.ops import control_flow_util

from model import Model

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 256, "Batch size.")


def main(_):
  dataset_train, info = tfds.load(name="squad/bytes",
                                  split=tfds.Split.TRAIN, with_info=True,
                                  data_dir="tensorflow_datasets",
                                  batch_size=FLAGS.batch_size)

  vocab_size = info.features["question"].encoder.vocab_size
  model = Model(vocab_size=vocab_size, emb_size=16, rnn_size=16)

  for i in range(1):
    for features in dataset_train:
      question = features["question"]
      output = model(question)
      logging.info(output.shape)


if __name__ == "__main__":
  app.run(main)

# model.py
import tensorflow as tf
from tensorflow.keras import layers


class Model(tf.keras.Model):
  def __init__(self, vocab_size, emb_size, rnn_size):
    super(Model, self).__init__()
    self.embedding = layers.Embedding(vocab_size, emb_size)
    self.rnn = DynamicRNN(rnn_size)

  def call(self, x):
    x = self.embedding(x)
    x, _ = self.rnn(x)
    return x

class DynamicRNN(layers.Layer):
  def __init__(self, rnn_size):
    super(DynamicRNNV1, self).__init__()
    self.rnn_size = rnn_size
    self.cell = layers.GRU(rnn_size, return_state=True)

  @tf.function
  def call(self, input_data):
    outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    state = tf.zeros((input_data.shape[0], self.rnn_size), dtype=tf.float32)
    for i in tf.range(input_data.shape[1]):
      print(input_data)
      output, state = self.cell(tf.expand_dims(input_data[:, i, :], 1), state)
      outputs = outputs.write(i, output)
    return tf.transpose(outputs.stack(), [1, 0, 2]), state

class DynamicRNNV2(layers.Layer):
  def __init__(self, rnn_size):
    super(DynamicRNNV2, self).__init__()
    self.rnn_size = rnn_size
    self.cell = layers.GRU(rnn_size, return_state=True)

  def call(self, input_data):
    state = tf.zeros((input_data.shape[0], self.rnn_size), dtype=tf.float32)
    outputs = []
    for i in range(input_data.shape[1]):
      output, state = self.cell(tf.expand_dims(input_data[:, i, :], 1), state)
      outputs.append(tf.expand_dims(output, 1))
    return tf.concat(outputs, axis=1), state

# train.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.ops import control_flow_util

from model import Model

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 256, "Batch size.")


def main(_):
  dataset_train, info = tfds.load(name="squad/bytes",
                                  split=tfds.Split.TRAIN, with_info=True,
                                  data_dir="tensorflow_datasets",
                                  batch_size=FLAGS.batch_size)

  vocab_size = info.features["question"].encoder.vocab_size
  model = Model(vocab_size=vocab_size, emb_size=16, rnn_size=16)

  for i in range(1):
    for features in dataset_train:
      question = features["question"]
      output = model(question)
      logging.info(output.shape)


if __name__ == "__main__":
  app.run(main)