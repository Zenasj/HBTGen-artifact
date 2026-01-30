from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import math
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


def data_generator(X, Y, batch_size, start=0, end=None):
    end = len(X) if end is None else end
    num_batches = int(math.ceil((end-start)/batch_size))
    while True:
        lob = list(range(num_batches))
        for bi in lob:
            sb = start + bi*batch_size
            eb = sb + batch_size
            eb = end if eb > end else eb
            Xb = X[sb:eb]
            Yb = Y[sb:eb]
            yield Xb, Yb


(Xtr, Ytr), (Xva, Yva) = tf.keras.datasets.cifar10.load_data()
Xtr, Ytr, Xva, Yva, nc = Xtr[:1000], Ytr[:1000], Xva[:100], Yva[:100], 10
Xtr, Xva = Xtr.astype('float32') / 255, Xva.astype('float32') / 255
Ytr, Yva, ins = to_categorical(Ytr, nc), to_categorical(Yva, nc), Xtr.shape[1:]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(ins))
model.add(tf.keras.layers.Conv2D(8, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
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
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[])

batch_size = 32
tr_gen = data_generator(Xtr, Ytr, batch_size)
va_gen = data_generator(Xva, Yva, batch_size)
tr_steps = int(math.ceil(len(Xtr)/batch_size))
va_steps = int(math.ceil(len(Xva)/batch_size))

model.fit_generator(tr_gen, steps_per_epoch=tr_steps, epochs=2, verbose=2,
                    validation_data=va_gen, validation_steps=va_steps)