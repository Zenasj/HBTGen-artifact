import random
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

model.save("test") #runs fine

optimizer = tf.keras.optimizers.SGD(),
model.compile(optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy())

model.save("test") #throws error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

x = np.random.random((128,28,28,1))
y = np.random.random((128, 10))

optimizer = tf.keras.optimizers.SGD(),
model.compile(optimizer, loss=tf.keras.losses.CategoricalCrossentropy())

model.fit(x=x, y=y, batch_size=32) #training fails

#Alternative way:
model.compile('sgd', loss=tf.keras.losses.CategoricalCrossentropy())
model.fit(x=x, y=y, batch_size=32) #training works