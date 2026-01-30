import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
batch_size = 100
ds = tf.data.Dataset.from_tensor_slices((np.random.normal(size=(10000,10)), np.zeros(10000))) \
.batch(batch_size) \
.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(256),
  tf.keras.layers.Dense(1)
])


optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False))
model.fit(ds,
          epochs=1)


model.save("test_model",save_format="tf")
converter = tf.lite.TFLiteConverter.from_saved_model('test_model',signature_keys=['serving_default'])
tflite_model = converter.convert()