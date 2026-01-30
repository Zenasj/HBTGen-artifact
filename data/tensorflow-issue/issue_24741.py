import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
keras = tf.keras

# tf.data.Dataset instance
tr_data = np.random.random((1000, 32)).astype(np.float32)
tr_label = np.random.randint(low=0, high=10, size = 1000).astype(np.int32)
tr_dataset = tf.data.Dataset.from_tensor_slices((tr_data, tr_label))
tr_dataset = tr_dataset.batch(batch_size=32)
tr_dataset = tr_dataset.repeat()

val_data = np.random.random((100, 32)).astype(np.float32)
val_label = np.random.randint(low=0, high=10, size = 100).astype(np.int32)
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_label))
val_dataset = val_dataset.batch(batch_size=100).repeat()

# Training
model = keras.Sequential()
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=10, activation='softmax'))
model.compile(optimizer=tf.train.GradientDescentOptimizer(.01), 
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(tr_dataset, epochs = 5, steps_per_epoch = 1000 // 32,
          validation_data = val_dataset, validation_steps = 1)

import numpy as np
import tensorflow as tf
keras = tf.keras

# np.array 
tr_data = np.random.random((1000, 32)).astype(np.float32)
tr_label = np.random.randint(low=0, high=10, size = 1000).astype(np.int32)
# tr_dataset = tf.data.Dataset.from_tensor_slices((tr_data, tr_label))
# tr_dataset = tr_dataset.batch(batch_size=32)
# tr_dataset = tr_dataset.repeat()

val_data = np.random.random((100, 32)).astype(np.float32)
val_label = np.random.randint(low=0, high=10, size = 100).astype(np.int32)
# val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_label))
# val_dataset = val_dataset.batch(batch_size=100).repeat()

# Training
model = keras.Sequential()
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=10, activation='softmax'))
model.compile(optimizer=tf.train.GradientDescentOptimizer(.01), 
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x=tr_data, y=tr_label, epochs=5, batch_size=32, validation_data=(val_data, val_label))
# model.fit(tr_dataset, epochs = 5, steps_per_epoch = 1000 // 32,
#           validation_data = val_dataset, validation_steps = 1)