import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.random.set_seed(2)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_dir = "data/train"
test_dir = "data/test"
total_classes = 200


# Define simple image model
model_input = Input((32, 32, 3))

x = Conv2D(128, (7, 7))(model_input)
x = Conv2D(64, (3, 3))(x)
x = Conv2D(32, (3, 3))(x)
x = Flatten()(x)
x = Dense(400, activation="relu")(x)
x = Dense(total_classes, activation="softmax")(x)

model = Model(inputs=model_input, outputs=x)
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Define data prep
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    color_mode='rgb',
    target_size=(32, 32),
    batch_size=128,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    color_mode='rgb',
    target_size=(32, 32),
    batch_size=128,
    class_mode='categorical')

model.fit(
    train_generator,
    validation_data=test_generator,
    validation_freq=1,
    epochs=10,
    workers=6,
    use_multiprocessing=True
)