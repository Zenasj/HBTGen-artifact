import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow import keras

data_a = np.array([300, 455, 350, 560, 700, 800, 200, 250], dtype=np.float32)
labels = np.array([455, 350, 560, 700, 800, 200, 250, 300], dtype=np.float32)
data_b = np.array([200, 255, 350, 470, 600, 300, 344, 322], dtype=np.float32)
data_a = np.reshape(data_a, (8, 1, 1))
data_b = np.reshape(data_b, (8, 1, 1))

x = keras.layers.Input(shape=(1, 1), name='input_x')
y = keras.layers.Input(shape=(1, 1), name='input_y')
admi = keras.layers.LSTM(40, return_sequences=False)(x)
pla = keras.layers.LSTM(40, return_sequences=False)(y)
out = keras.layers.concatenate([admi, pla], axis=-1)
output = keras.layers.Dense(1, activation='sigmoid')(out)
model = keras.models.Model(inputs=[x, y], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit([data_a, data_b], labels, batch_size=2, epochs=10)
model.fit({'input_x': data_a, 'input_y': data_b}, labels, batch_size=2, epochs=10)

dataset = tf.data.Dataset.from_tensor_slices(({'input_x': data_a, 'input_y': data_b}, labels)).batch(2).repeat()
model.fit(dataset, epochs=10, steps_per_epoch=4)

def generator():
    while True:
        for i in np.random.permutation(8):
            yield {'input_x': data_a[i], 'input_y': data_b[i]}, labels[i]

dataset = tf.data.Dataset.from_generator(generator, ({'input_x': tf.float32, 'input_y': tf.float32}, tf.float32)).batch(2)
model.fit(dataset, epochs=10, steps_per_epoch=4)

dataset = tf.data.Dataset.from_tensor_slices(((data_a, data_b), labels)).batch(2).repeat()
model.fit(dataset, epochs=10, steps_per_epoch=4)

def generator():
    while True:
        for i in np.random.permutation(8):
            yield (data_a[i], data_b[i]), labels[i]

dataset = tf.data.Dataset.from_generator(generator, ((tf.float32, tf.float32), tf.float32)).batch(2)
model.fit(dataset, epochs=10, steps_per_epoch=4)

xy_ds = (
        tf.data.Dataset.zip((audio_ds, label_ds))
            .batch(
            batch_size=batch_size,
            # drop_remainder=True if is_training else False
            )
        .repeat(repeat)
        .prefetch(tf.contrib.data.AUTOTUNE)
    )

model = Model(inputs=[input1, input2], outputs=predictions)