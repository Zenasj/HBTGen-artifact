import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

# establish connection to TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)
print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  

with strategy.scope():
  # define model layers
  model_input = tf.keras.layers.Input(shape=(224, 224, 1))
  x = tf.keras.layers.experimental.preprocessing.RandomRotation((-0.1, 0.1))(model_input)
  x = tf.keras.layers.MaxPool2D((224, 224))(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

  # compile model
  model = tf.keras.Model(inputs=model_input, outputs=x)
  model.summary(line_length=120)
  model.compile(optimizer='adam', loss='binary_crossentropy')

  # generate some random data and fit the model
  images = tf.random.uniform((10, 224, 224, 1))
  labels = tf.zeros((10, 1))
  model.fit(images, labels)

  # predict on the data
  print(model.predict(images))

augmentor = tf.keras.layers.experimental.preprocessing.RandomRotation((-0.1, 0.1))
ds = ds.map(lambda x, y: (augmentor.call(x), y))