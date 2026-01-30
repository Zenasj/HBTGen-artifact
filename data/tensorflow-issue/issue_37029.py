import random
from tensorflow import keras

import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np


def random_image_generator(batch_size, num_classes, input_shape):
    templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
    random_data = np.random.normal(loc=0, scale=1., size=input_shape)
    while True:
        y = np.random.randint(0, num_classes, size=(batch_size,))
        x = np.zeros((batch_size,) + input_shape, dtype=np.float32)
        for i in range(batch_size):
            x[i] = templates[y[i]] + random_data
        x_array = np.array(x)
        y_array = tf.keras.utils.to_categorical(y, num_classes)
        yield(x_array, y_array)

def run_model():
    K.set_image_data_format('channels_first')
    image_dim = 5000
    input_shape = (3, image_dim, image_dim)

    num_classes = 15
    batch_size = 1
    model_class = tf.keras.applications.ResNet50
    model = model_class(weights=None, include_top=True, input_shape=input_shape,
                        classes=num_classes)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    random_generator = random_image_generator(batch_size, num_classes,
                                              input_shape)
    model.fit(random_generator, steps_per_epoch=10,
              epochs=1)

run_model()

import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np

image_dim = 5000
input_shape = (3, image_dim, image_dim)

num_classes = 15
batch_size = 1

templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
random_data = np.random.normal(loc=0, scale=1., size=input_shape)
y = np.random.randint(0, num_classes, size=(batch_size,))
x = np.zeros((batch_size,) + input_shape, dtype=np.float32)
for i in range(batch_size):
    x[i] = templates[y[i]] + random_data
x_array = np.array(x)
y_array = tf.keras.utils.to_categorical(y, num_classes)


def random_image_generator(batch_size, num_classes, input_shape):
    while True:
        yield(x_array, y_array)

def run_model():
    K.set_image_data_format('channels_first')

    model_class = tf.keras.applications.ResNet50
    model = model_class(weights=None, include_top=True, input_shape=input_shape,
                        classes=num_classes)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    random_generator = random_image_generator(batch_size, num_classes,
                                              input_shape)
    model.fit(random_generator, steps_per_epoch=10,
              epochs=1)

run_model()

def random_image_generator(batch_size, num_classes, input_shape):
    templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
    random_data = np.random.normal(loc=0, scale=1., size=input_shape)
    y = np.random.randint(0, num_classes, size=(batch_size,))
    x = np.zeros((batch_size,) + input_shape, dtype=np.float32)
    for i in range(batch_size):
        x[i] = templates[y[i]] + random_data
    x_array = np.array(x)
    y_array = tf.keras.utils.to_categorical(y, num_classes)
    while True:
        print('random_image_generator while start')
        time.sleep(1)
        print('random_image_generator while yield')
        yield(x_array, y_array)