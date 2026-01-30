from tensorflow import keras
from tensorflow.keras import layers

import os
import tensorflow as tf
import numpy as np

tf.get_logger().propagate = False
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu = 'grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():

  def encode_jpg(image, class_idx):
    return tf.io.encode_jpeg(image, quality = 95, optimize_size = True, chroma_downsampling = False), class_idx

  def decode_jpg(image, class_idx):
    return tf.image.decode_jpeg(image, channels = 3), class_idx

  def normalize_img(image, class_idx):
    return image / 255 - 0.5, class_idx

  dataset = tf.data.Dataset.from_tensor_slices((
    [tf.cast(np.zeros((256, 256, 3)), dtype = tf.uint8) for _ in range(300)],
    [0 for _ in range(300)]
  ))
  dataset = dataset.map(encode_jpg)
  dataset = dataset.map(decode_jpg)
  dataset = dataset.map(normalize_img)
  dataset = dataset.batch(8)

  print('\nhow does our dataset look like?')
  for i, (image, label) in enumerate(dataset):
    print(image.shape, label.shape)
    if i == 2: break

  model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (256, 256, 3)),
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(10)
  ])

  print('\nhow does our model model like?')
  model.summary()

  model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam')
  model.fit(dataset, epochs = 1)

dataset = dataset.map(encode_jpg)
dataset = dataset.map(decode_jpg)

def decode_jpg(image, class_idx):
  return tf.reshape(tf.image.decode_jpeg(image, channels = 3),[256,256, 3]), class_idx