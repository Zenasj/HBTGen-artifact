from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, BatchNormalization,
                                     MaxPool2D, Flatten, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.image import random_hue
import numpy as np

# random hue augmentations - PROBLEMATIC WITH MULTIPROCESSING
def color_augmentation(image):
    return random_hue(image, 0.1)

# get the data generators
# apply random augmentations during training including hue augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    vertical_flip = True,
    horizontal_flip = True,
    preprocessing_function = color_augmentation
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train/',
    target_size = (96, 96),
    batch_size = 32,
)

test_generator = test_datagen.flow_from_directory(
    'test/',
    target_size = (96, 96),
    batch_size = 32,
)

# create model
model = Sequential()
model.add(Conv2D(filters=32, strides=1,input_shape=(96,96,3),
                 activation='relu', kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Conv2D(filters=64, strides=1, activation='relu',
                 kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(units=2, activation='softmax'))

# compile model
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# fit the model, using multiprocessing
history = model.fit(
    x = train_generator,
    steps_per_epoch = len(train_generator),
    epochs = 5,
    verbose = 1,
    validation_data = test_generator,
    validation_steps = len(test_generator),
    workers = 4,
    use_multiprocessing = True,
    max_queue_size = 8
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, BatchNormalization,
                                     MaxPool2D, Flatten, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.image import random_hue
from tensorflow.keras.datasets import cifar10
import numpy as np


# random hue augmentations - PROBLEMATIC WITH MULTIPROCESSING
def color_augmentation(image):
    return random_hue(image, 0.1)

# get the datagenerators. Apply random augmentations during training including
# random hue augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    vertical_flip = True,
    horizontal_flip = True,
    preprocessing_function = color_augmentation
)

test_datagen = ImageDataGenerator(rescale=1./255)

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

train_generator = train_datagen.flow(
     x_train, y=y_train, batch_size=32
)

test_generator = test_datagen.flow(
    x_test, y=y_test, batch_size=32
)

# create model
model = Sequential()
model.add(Conv2D(filters=32, strides=1,input_shape=(32,32,3),
                 activation='relu', kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Conv2D(filters=64, strides=1, activation='relu',
                 kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))

# compile model
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# fit the model, using multiprocessing
history = model.fit(
    x = train_generator,
    steps_per_epoch = len(train_generator),
    epochs = 5,
    verbose = 1,
    validation_data = test_generator,
    validation_steps = len(test_generator),
    workers = 4,
    use_multiprocessing = True,
    max_queue_size = 8
)