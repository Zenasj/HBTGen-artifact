import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, Dense, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Model


X_train = np.random.random((235,65, 320, 1))
y_train = np.random.random((235, 65, 320, 1))

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data = dataset.shuffle(len(X_train)).batch(32)
train_data = train_data.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

input_shape = (65, 320, 1)
initializer = tf.keras.initializers.HeUniform()

model = tf.keras.Sequential(
[
    tf.keras.Input(shape=(65, 320, 1)),
    keras.layers.Rescaling(1./125,offset=1., input_shape=(65,320,1)),
    keras.layers.Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer=initializer),
    keras.layers.Dense(20, use_bias=True),
    keras.layers.Dropout(rate=0.7),
    keras.layers.Dense(1, activation='linear')
])

model.summary()
lossMse = MeanSquaredError()
model.compile(optimizer='adam',
                loss=lossMse,
                metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, verbose=1)