import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

ones = tf.ones((1, 10000, 1000))
zeros = tf.zeros((1, 10000, 1000))
data = tf.data.Dataset.from_tensor_slices(
    (
        ones,
        tf.ones((1, 1))
    )
).concatenate(
    tf.data.Dataset.from_tensor_slices(
        (
            zeros,
            tf.zeros((1, 1))
        )
    )
).map(lambda x, y: (tf.image.random_crop(x, (20, 20)), y)).repeat().batch(10)

x = tf.keras.layers.Input((20, 20))
y = tf.keras.layers.Dense(1, tf.keras.activations.sigmoid)(x)
model = tf.keras.models.Model(inputs=[x], outputs=[y])
model.compile(loss='mean_squared_error', optimizer='ADAM')
model.fit(x=data, epochs=100, steps_per_epoch=100, validation_data=validation, validation_steps=3)