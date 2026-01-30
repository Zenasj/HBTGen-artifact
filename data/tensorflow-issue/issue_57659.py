from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
print(tf.config.optimizer.get_experimental_options())
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
import shutil

import keras.backend
import numpy as np
import pandas
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("svg")

IMG_SIZE = 224
CHANNELS = 3
BATCH_SIZE = 1
LR_2 = 0.003
EPOCHS = 2

class VGG19(Sequential):
    def __init__(self, labels, input_shape):
        super().__init__()

        self.add(Conv2D(64, kernel_size=(3, 3), padding='same',
                        activation='relu', input_shape=input_shape))
        self.add(Conv2D(64, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(Conv2D(128, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(Conv2D(128, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(Conv2D(256, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(Conv2D(256, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(Conv2D(256, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(Conv2D(256, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same',
                        activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(Flatten())
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(labels, activation='sigmoid'))

        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_2, decay=LR_2 / EPOCHS),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

model = VGG19(labels, (IMG_SIZE, IMG_SIZE, CHANNELS))
# training and validation is an assortment of images augmented by ImageDataGenerator

H = model.fit(train_generator,
              steps_per_epoch=STEP_SIZE_TRAIN,
              validation_data=validate_generator,
              validation_steps=STEP_SIZE_VALID,
              epochs=classify.EPOCHS,
              verbose=1
              )

model.save('data/model.h5', save_format="h5")