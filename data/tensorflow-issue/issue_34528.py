from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

def loss_func(y_true, y_pred):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f, axis=1, keepdims=True)
    union = K.sum(y_true_f, axis=1, keepdims=True) \
            + K.sum(y_pred_f, axis=1, keepdims=True)
    return -(2. * intersection + K.epsilon()) / (union + K.epsilon())

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
model.compile(loss=loss_func, optimizer=opt, metrics=[])

Ytr, Yva = Ytr.astype('uint8'), Yva.astype('uint8') # (1)

model.fit(x=Xtr, y=Ytr, batch_size=32, epochs=1, validation_data=(Xva, Yva))