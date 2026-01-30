import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *

tf.get_logger().propagate = False
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu = 'grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
#with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):

  xception = tf.keras.applications.Xception(weights = 'imagenet', include_top = False, input_shape = (256, 256, 3))

  x = xception.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(256, activation = 'relu')(x)
  x = Dropout(0.25)(x)
  predictions = Dense(10)(x)

  model = tf.keras.Model(inputs = xception.input, outputs = predictions)

  model.compile(
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
      optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-8),
      metrics = ['accuracy']
  )

  batch_size = 7
  for epoch in range(5):
    X = np.random.rand(batch_size, 256, 256, 3)
    Y = np.zeros(batch_size)
    model.fit(X, Y, batch_size = batch_size, epochs = 1)

for layer in xception.layers:
  layer.trainable = False