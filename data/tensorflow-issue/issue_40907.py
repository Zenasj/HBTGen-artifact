from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import ModelCheckpoint



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

def get_new_model():
    model = Sequential([
        Conv2D(filters=16, input_shape=(32, 32, 3), kernel_size=(3, 3), 
               activation='relu', name='conv_1'),
        tf.keras.layers.BatchNormalization(),
        Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='conv_2'),
        MaxPooling2D(pool_size=(4, 4), name='pool_1'),
        tf.keras.layers.BatchNormalization(),
        Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='conv_3'),
        MaxPooling2D(pool_size=(4, 4), name='pool_2'),
        Flatten(name='flatten'),
        Dense(units=32, activation='relu', name='dense_1'),
        tf.keras.layers.Dropout(0.5),
        Dense(units=10, activation='softmax', name='dense_2')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


checkpoint_5000_path = 'checkpoint_5000/checkpoint_{epoch:02d}_{batch:04d}'

model = get_new_model()
checkpoint_5000 = ModelCheckpoint(filepath=checkpoint_5000_path, verbose=True, save_weights_only=True,
                                  save_freq=5000)
model.fit(x_train, y_train, batch_size=10, validation_data=(x_test,y_test), epochs=3, verbose= True, callbacks=[checkpoint_5000])