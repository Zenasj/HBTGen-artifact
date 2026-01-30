import random
from tensorflow import keras
from tensorflow.keras import layers

3
import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

os.environ["TFHUB_CACHE_DIR"] = '/tmp/tfhub'

# Create model
model = tf.keras.Sequential([tf.keras.layers.TimeDistributed(hub.KerasLayer("https://tfhub.dev/google/bit/s-r101x1/1",
                                                                            trainable=False)),
                             tf.keras.layers.LSTM(units=512),
                             tf.keras.layers.Dense(units=128,
                                                   activation=tf.nn.relu),
                             tf.keras.layers.Dense(units=1,
                                                   activation=tf.nn.sigmoid)])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Fit
model.fit(x=np.random.randint(low=0, high=256, size=(1000, 10, 56, 56, 3)).astype(float),
          y=np.random.randint(low=0, high=2, size=(1000,)).astype(float), epochs=1)