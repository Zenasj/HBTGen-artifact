from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize to between 0-1

# model layers
xIn = Input((28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu')(xIn)
x = Dropout(0.4)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Dropout(0.4)(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = Dropout(0.4)(x)
x = Flatten()(x)
x = Dense(128, activation='swish')(x)
x = Dropout(0.5)(x)
xOut = Dense(10)(x)

model = Model(inputs=xIn, outputs=xOut)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, verbose=1)
]

model.summary()
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_data=(x_test, y_test), callbacks=callbacks)