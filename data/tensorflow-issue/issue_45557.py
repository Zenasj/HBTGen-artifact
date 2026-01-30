import random
from tensorflow import keras
from tensorflow.keras import layers

# Train a small model and save it as a SavedModel
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dense(2))
model.compile(optimizer=tfa.optimizers.NovoGrad(), loss=tf.keras.losses.sparse_categorical_crossentropy)
callbacks = [tf.keras.callbacks.ModelCheckpoint("scratch")]

x = np.random.random((2, 3))
y = np.random.randint(0, 2, (2,))
model.fit(x, y, batch_size=1, epochs=1, callbacks=callbacks)