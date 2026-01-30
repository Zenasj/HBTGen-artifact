import random
from tensorflow import keras
from tensorflow.keras import layers

#!pip install -q tensorflow==1.13.1
import tensorflow as tf
import numpy as np
import os

print(tf.__version__)

tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']

randomImages = np.random.rand(128,50,50,3)
randomLabels = np.random.randint(2, size=128)

#model
inputs = tf.keras.layers.Input(shape=(50, 50, 3))
outputLayer = inputs
for i in range(0,3):
  outputLayer = tf.keras.layers.Conv2D(filters=64, kernel_size = 3, padding = 'valid')(outputLayer)
  outputLayer = tf.keras.layers.BatchNormalization()(outputLayer)
  outputLayer = tf.keras.layers.ReLU()(outputLayer)
outputLayer = tf.keras.layers.Flatten()(outputLayer)
outputLayer = tf.keras.layers.Dense(2)(outputLayer)

model = tf.keras.Model(inputs = inputs, outputs = outputLayer)

#tpu_model 
tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
    tf.contrib.cluster_resolver.TPUClusterResolver(tpu_address))
  )

tpu_model.compile(
    optimizer=tf.train.AdamOptimizer(learning_rate = 0.0001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['sparse_categorical_accuracy']
  )


tpu_model.fit(
              x = randomImages,
              y = randomLabels,
              batch_size = 64,
              epochs = 10
            )