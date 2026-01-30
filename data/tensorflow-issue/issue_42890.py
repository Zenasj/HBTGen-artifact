from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

#!pip install tensorflow==2.2.0
import tensorflow as tf
import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import keras_preprocessing
from keras_preprocessing import image

from tensorflow.python.keras.utils.version_utils import training
from tensorflow.keras.optimizers import RMSprop

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

x_trainf = train_images.astype('float32') / 255.0
x_testf = test_images.astype('float32') / 255.0

x_train_r = x_trainf.reshape(x_trainf.shape[0], 28, 28, 1)
x_test_r = x_testf.reshape(x_testf.shape[0], 28, 28, 1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train_r, 
          train_labels, batch_size=32,  
          epochs=10,
          validation_data=(x_test_r,test_labels))

model.evaluate(x_test_r,test_labels)

model.save('/tmp/loaddatamnist.h5')

loadedmodel=tf.keras.models.load_model('/tmp/loaddatamnist.h5')
loadedmodel.evaluate(x_test_r,test_labels)

loadedmodel=tf.keras.models.load_model('/tmp/test.h5')
loadedmodel.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
loadedmodel.evaluate(x_test_r,test_labels)

model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])