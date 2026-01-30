import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

a = np.random.rand(512, 512, 3)
b = np.random.rand(1080, 1920, 3)

values = tf.ragged.stack([a, b])
labels = np.asarray([0, 1])

print(values.shape)
# NHWC format
# TensorShape([2, None, None, 3])

print(labels.shape)
# (2,)

# Highly simplified image model
model = tf.keras.models.Sequential()
# The output of this layer will always be (Batch, 224, 224, 3), so adding support for
# ragged tensors shouldn't require updates for downstream ops
model.add(
  tf.keras.layers.experimental.preprocessing.Resizing(
    224,
    224,
    input_shape=(None, None, 3), 
    name="resize"))
model.add(tf.keras.layers.Conv2D(kernel_size=3, filters=24, name="kernel"))
model.add(tf.keras.layers.GlobalMaxPool2D(name="pool"))
model.add(tf.keras.layers.Dense(1, name="dense_second"))

model.compile()

model.fit(values, labels)