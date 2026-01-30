from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

foo = tf.Variable(3.0)
ema = tf.train.ExponentialMovingAverage(0.1)
decayed_foo = ema.apply([foo])

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.expand_dims(x_train, -1) / 255.0, np.expand_dims(x_test, -1) / 255.0


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, (3, 3),
            kernel_initializer=tf.keras.initializers.he_uniform(),
            activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            32, (3, 3),
            (2, 2),
            kernel_initializer=tf.keras.initializers.he_uniform(),
            activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            32, (3, 3),
            kernel_initializer=tf.keras.initializers.he_uniform(),
            activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            64, (3, 3),
            (2, 2),
            kernel_initializer=tf.keras.initializers.he_uniform(),
            activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            64, (3, 3),
            kernel_initializer=tf.keras.initializers.he_uniform(),
            activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(
            64, kernel_initializer=tf.keras.initializers.he_uniform(),
            activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(
            10, kernel_initializer=tf.keras.initializers.he_uniform(),
            activation='softmax')
    ]
)

optimizer = tfa.optimizers.LazyAdam(decay=1e-4)
optimizer = tfa.optimizers.MovingAverage(optimizer)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=10)
print("Test metrics before mae:")
model.evaluate(x_test, y_test)

optimizer.assign_average_vars(model.variables)
print("Test metrics after mae:")
model.evaluate(x_test, y_test)
model.save('./model.h5')