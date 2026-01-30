import random
from tensorflow import keras
from tensorflow.keras import layers

import os
import time
import numpy as np
import tensorflow as tf

tf.keras.backend.clear_session()

TF_MASTER = 'grpc://{}'.format(os.environ['COLAB_TPU_ADDR'])

with tf.Session(TF_MASTER) as session:
  print(session.list_devices())

print("Tensorflow Version:", tf.VERSION)
print("Tensorflow Keras Version:", tf.keras.__version__)

amount = 128
size = [256, 256]
images = np.array([np.random.rand(*size, 3).astype('float32') for i in range(amount)])
masks = np.array([np.random.rand(*size, 1).astype('float32') for i in range(amount)])

ds = tf.data.Dataset.from_tensor_slices((images, masks)).batch(amount, drop_remainder=True) # we need Dataset to run on TPU
print(ds)

# very minimal model to demostrate the issue
def make_model(batch_size=None):
  src = tf.keras.layers.Input(shape=(*size,3), batch_size=batch_size, dtype=tf.float32, name='Input')
  outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(src)

  model = tf.keras.Model(inputs=[src], outputs=[outputs])
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model

resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.contrib.distribute.initialize_tpu_system(resolver)
strategy = tf.contrib.distribute.TPUStrategy(resolver)
with strategy.scope():
  model = make_model(batch_size = 128)
model.summary()

for i in range(30):
  start_time = time.time()
  model.evaluate(ds, steps=1)
  print("--- %s seconds ---" % (time.time() - start_time))