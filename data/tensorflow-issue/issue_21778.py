from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

max_pool = tf.keras.layers.MaxPooling2D(2)
model = tf.keras.Sequential([
    max_pool,
    max_pool,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')

X, y = np.zeros((32, 8, 8, 1)), np.ones((32,))
model.fit(X, y)

tf.keras.estimator.model_to_estimator(keras_model=model)

import tensorflow as tf
import numpy as np

max_pool = tf.keras.layers.MaxPooling2D(2)
model = tf.keras.Sequential([
    max_pool,
    max_pool,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy')

X, y = np.zeros((32, 8, 8, 1)), np.ones((32, 1))
model.fit(X, y)

estimator = tf.keras.estimator.model_to_estimator(keras_model=model)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x=X,
    y=y,
    batch_size=32,
    num_epochs=1,
    shuffle=False
)

estimator.train(input_fn=input_fn, steps=1)

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH),
            filters=96, kernel_size=11, strides=4, padding='valid'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='valid'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='valid'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=4096),
        tf.keras.layers.Dense(units=4096),
        tf.keras.layers.Dense(units=num_classes),
    ])