import random

import numpy as np
from tensorflow.keras import layers, optimizers, losses, Model, Input

inputs = Input(shape=(10,), name='img_input')

x1 = layers.Dense(5)(inputs)
x2 = layers.Dense(2)(inputs)

model = Model(inputs=inputs,
                    outputs=[x1, x2])

model.compile(
    optimizer=optimizers.Adam(),
    loss=losses.categorical_crossentropy)


img_data = np.random.random_sample(size=(1, 10))
targets_0 = np.random.random_sample(size=(1, 5))
targets_1 = np.random.random_sample(size=(1, 2))

def generator_tuple():

    while True:
        yield img_data, (targets_0, targets_1)


def generator_list():
    while True:
        yield img_data, [targets_0, targets_1]

model.fit_generator(generator_tuple(), steps_per_epoch=1, epochs=3) # ok
model.fit(generator_tuple(), steps_per_epoch=1, epochs=3) # ok
model.fit_generator(generator_list(), steps_per_epoch=1, epochs=3) # ok
model.fit(generator_list(), steps_per_epoch=1, epochs=3) # raise error