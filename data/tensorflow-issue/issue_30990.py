from tensorflow import keras

import tensorflow as tf

from tensorflow.python.keras import layers
from tensorflow.python.keras.callbacks import ModelCheckpoint

model = tf.keras.Sequential()
model.add(layers.LSTM(units=64, input_shape=(28, 28), return_sequences=False))
model.add(layers.Dense(10, activation='softmax'))

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample, sample_label = x_train[0], y_train[0]

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=[])

callback = ModelCheckpoint(filepath='saved/',
                           monitor='val_loss',
                           save_weights_only=False,
                           mode='min', save_freq='epoch')

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=64, epochs=2, callbacks=[callback])