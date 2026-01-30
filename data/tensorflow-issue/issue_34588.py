from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D

ds_train = tfds.load(name="cifar100", split=tfds.Split.TRAIN)
ds_test = tfds.load(name="cifar100", split=tfds.Split.TEST)

input_shape = (32,32, 3)
num_classes = 10
epochs = 10

model = tf.keras.Sequential([
    Conv2D(32, 5, padding='same', activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2), (2, 2), padding='same'),
    BatchNormalization(),
    Conv2D(64, 5, padding='same', activation='relu'),
    MaxPooling2D((2, 2), (2, 2), padding='same'),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

model.fit(ds_train,
          epochs=epochs,
          validation_data=ds_test,
          verbose=1)