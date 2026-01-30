from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, ReLU
from tensorflow.keras.layers import LayerNormalization, GlobalAveragePooling1D, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
  model = Sequential()
  model.add(LSTM(units=50, return_sequences=True))
  model.add(LayerNormalization())
  model.add(GlobalAveragePooling1D(data_format='channels_last'))
  model.add(Dense(7, activation='softmax'))
  model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                experimental_steps_per_execution=100,
                metrics=['accuracy'])

x=-np.ones((150, 100, 4)).astype(np.float32)
y=np.ones((150, 7)).astype(np.float32)
model.fit(x, y, epochs=10)

x=-np.ones((32, 100, 4)).astype(np.float32)
y=np.ones((32, 7)).astype(np.float32)