import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
session_config.allow_soft_placement = True
strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
config = tf.estimator.RunConfig(session_config=session_config, train_distribute=strategy)
estimator = tf.estimator.Estimator(model_fn, model_dir, config, params)
estimator.train()

import os
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np


def model_fn(features, labels, mode):
  """A simple 2-classify model.
  
  """
  model = tf.keras.Sequential([layers.Dense(80, activation="relu"), layers.Dense(2)])
  logits = model(features)
  loss = tf.losses.softmax_cross_entropy(labels, logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def input_fn():
  """dataset that return feature and label.

  """
  features_mat = np.random.randn(10, 10) 
  labels_mat = np.random.randint(0, 2, size=(10, 2)) 
  dataset = tf.data.Dataset.from_tensor_slices((features_mat, labels_mat))
  return dataset.batch(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set available GPU id. 
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True

session_config.allow_soft_placement = True
strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=1)
config = tf.estimator.RunConfig(session_config=session_config, train_distribute=strategy)
# If disable above 3 lines and using following line, GPU memory allocation will be correct.
#config = tf.estimator.RunConfig(session_config=session_config)

estimator = tf.estimator.Estimator(model_fn, config=config)

while True:
  estimator.train(input_fn)