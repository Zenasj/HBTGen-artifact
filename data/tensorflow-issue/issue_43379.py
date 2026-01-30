from tensorflow import keras

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

inp = layers.Input((1))
out1 = layers.Dense(32)(inp)
out2 = layers.Dense(32)(inp)

model = models.Model(inputs=[inp], outputs=[out1, out2])
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy())

def gen():
    for i in itertools.count(1):
        yield [1, 2], [[11, 12], [21, 22]]

generator_dataset = tf.data.Dataset.from_generator(
    gen,
    (tf.uint8, tf.uint8),
    output_shapes=(
        tf.TensorShape((2, 1)),
        tf.TensorShape((2, 2))
    )
)

model.fit(generator_dataset)