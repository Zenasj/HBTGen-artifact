import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


# Define the model using the Functional API
x = tf.keras.Input(shape=(20,), dtype=tf.float32)
y = tf.keras.Input(shape=(), dtype=tf.float32)
tmp = Dense(64, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(tmp)


class DummyLossLayer(tf.keras.layers.Layer):
    def call(self, *x):
        self.add_loss(tf.keras.losses.BinaryCrossentropy(from_logits=True)(*x))
        return x

outputs, _ = DummyLossLayer()(outputs, y)
model = Model(inputs=[x, y], outputs=outputs)


# Compile the model with the custom metric
model.compile(optimizer='adam')

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="cp-{epoch}",
    verbose=1,
    save_weights_only=True,
    save_freq='epoch')


# Dummy data for demonstration
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000,)).astype(np.float32)

# Train the model
model.fit([x_train, y_train], epochs=2, batch_size=32, callbacks=[cp_callback])