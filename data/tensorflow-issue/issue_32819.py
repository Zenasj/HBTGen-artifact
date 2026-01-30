from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

tf.enable_eager_execution()

def generator():
    for _ in range(100):
        yield [1, 1], 1

training = tf.data.Dataset \
    .from_generator(
        generator=generator,
        output_types=(tf.float64, tf.float64),
        output_shapes=(tf.TensorShape([2]), tf.TensorShape([])),
    ) \
    .batch(2) \
    .repeat()

validation = tf.data.Dataset \
    .from_generator(
        generator=generator,
        output_types=(tf.float64, tf.float64),
        output_shapes=(tf.TensorShape([2]), tf.TensorShape([])),
    ) \
    .batch(2)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(2),
    tf.keras.layers.Dense(1),
])
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(
    x=training,
    validation_data=validation,
    epochs=10,
    steps_per_epoch=20,
)