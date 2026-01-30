from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import h5py
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, HDF5Matrix

(Xtr, Ytr), (Xva, Yva) = tf.keras.datasets.cifar10.load_data()
Xtr, Ytr, Xva, Yva, nc = Xtr[:1000], Ytr[:1000], Xva[:100], Yva[:100], 10
Xtr, Xva = Xtr.astype('float32') / 255, Xva.astype('float32') / 255
Ytr, Yva, ins = to_categorical(Ytr, nc), to_categorical(Yva, nc), Xtr.shape[1:]
with h5py.File('cifar10.hdf5', 'w') as h5f:
    h5f['Xtr'], h5f['Ytr'], h5f['Xva'], h5f['Yva'] = Xtr, Ytr, Xva, Yva
del Xtr, Ytr, Xva, Yva

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(8, (3, 3), input_shape=ins, activation='relu'))
model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(nc, activation='softmax'))
opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

Xtr, Ytr = HDF5Matrix('cifar10.hdf5', 'Xtr'), HDF5Matrix('cifar10.hdf5', 'Ytr')
Xva, Yva = HDF5Matrix('cifar10.hdf5', 'Xva'), HDF5Matrix('cifar10.hdf5', 'Yva')

model.fit(x=Xtr, y=Ytr, batch_size=32, epochs=100,
          validation_data=(Xva, Yva), shuffle='batch')