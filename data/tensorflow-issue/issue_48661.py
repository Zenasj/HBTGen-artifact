from tensorflow.keras import layers
from tensorflow.keras import optimizers

class HDF5Generator(tf.keras.utils.Sequence):
    def __init__(self, hdf5_file):
        print("GENERATED")
        self.hdf5 = h5py.File(hdf5_file, 'r')
        self.indices = list(range(0, len(self.hdf5["samples"])))
        random.Random().shuffle(self.indices)

    def __len__(self):
        return len(self.hdf5["samples"])

    def __getitem__(self, idx):
        return self.hdf5["samples"][self.indices[idx]], self.hdf5["labels"][self.indices[idx]]

    def on_epoch_end(self):
        print("SHUFFLE")
        random.Random().shuffle(self.indices)

d = tf.data.Dataset.from_generator(HDF5Generator, args=[dataset], output_signature=(...))
d = d.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
model.fit(d, epochs=epochs)

import tensorflow as tf
import numpy as np
import random
from tensorflow.keras import models
from tensorflow.python.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow import keras

class HDF5Generator(tf.keras.utils.Sequence):
    def __init__(self):
        print("GENERATED")
        self.samples = np.random.rand(10000,5,20)
        self.labels = np.random.randint(2, size=(10000, 1))
        self.indices = list(range(0, 10000))
        random.Random().shuffle(self.indices)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        return self.samples[self.indices[idx]], self.labels[self.indices[idx]]

    def on_epoch_end(self):
        print("SHUFFLE")
        raise Exception("AAA")
        random.Random().shuffle(self.indices)

d = tf.data.Dataset.from_generator(HDF5Generator, output_signature=(tf.TensorSpec(shape=(5,20)), tf.TensorSpec(shape=(1,))))
d = d.batch(32).prefetch(tf.data.AUTOTUNE).cache()

model = models.Sequential()
model.add(Conv1D(32, kernel_size=1, strides=1, activation="relu", padding="same", input_shape=(5,20)))
model.add(MaxPooling1D(pool_size=3, padding="same"))
model.add(Conv1D(64, kernel_size=1, strides=1, activation="relu", padding="same", input_shape=(5,20)))
model.add(MaxPooling1D(pool_size=3, padding="same"))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dense(1, activation="softmax"))
model.compile(
        optimizer=keras.optimizers.Adam(0.003),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
model.fit(d, epochs=5)

class CallbackOnEpochEnd(Callback):
    def __init__(self, generator):
        super(CallbackOnEpochEnd, self).__init__()
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        self.generator.on_epoch_end()

[...]

generator = HDF5Generator()
d = tf.data.Dataset.from_generator(lambda: generator, output_signature=(tf.TensorSpec(shape=(5,20)), tf.TensorSpec(shape=(1,))))

[...]

on_epoch_end_callback = CallbackOnEpochEnd(generator)

[...]

model.fit(d, epochs=5, callbacks=[on_epoch_end_callback])