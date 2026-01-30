from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import itertools
import tensorflow as tf
import tensorflow.keras.backend as K

config = tf.ConfigProto(device_count={'GPU': 0})
K.set_session(tf.Session(config=config))

def train_fn():
    i = tf.keras.layers.Input(shape=(1,))
    x = tf.keras.layers.Dense(100, activation='relu')(i)
    o = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(i, o)
    model.compile('adam', 'mae')

    def gen():
        for i in itertools.count(1):
            yield [float(i)], [float(i)]

    ds = tf.data.Dataset.from_generator(
        gen, (tf.float32, tf.float32), (tf.TensorShape([1]), tf.TensorShape([1])))

    model.fit(ds, steps_per_epoch=10)

train_fn()

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

train_path = os.path.join(os.path.dirname(__file__), "./CNN Data/Training Set")
valid_path = os.path.join(os.path.dirname(__file__), "./CNN Data/Validation Set")
test_path = os.path.join(os.path.dirname(__file__), "./CNN Data/Testing Set")

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(251,388), classes=['Cortex','Medulla', 'Pelvis'], batch_size=6)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(251,388), classes=['Cortex','Medulla', 'Pelvis'], batch_size=7)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(251,388), classes=['Cortex','Medulla', 'Pelvis'], batch_size=7)

model = keras.Sequential([
    Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same', input_shape=(251,388, 1)),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),
    Flatten(),
    Dense(3, activation='softmax')
])

EPOCHS = 20
INIT_LR = 0.001

model.compile(Adam(learning_rate=INIT_LR), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_batches, steps_per_epoch=36, epochs=EPOCHS, verbose=1,
    validation_data=valid_batches, validation_steps=18
)