import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from collections import OrderedDict
import tensorflow as tf


batch_size = 3
input_values = (
    OrderedDict((
        ('a', tf.random.uniform((batch_size, 1))),
        ('b', tf.random.uniform((batch_size, 4))),
    )),
    (
        tf.random.uniform((batch_size, 3)),
        tf.random.uniform((batch_size, 4)),
    ),
)

inputs = tf.nest.map_structure(
    lambda x: tf.keras.layers.Input(tf.shape(x)[1:]), input_values)


def create_model(inputs):
    sequential_model = tf.keras.Sequential((
        tf.keras.layers.Lambda(
            lambda x: tf.concat(tf.nest.flatten(x), axis=-1)),
    ))
    out = sequential_model(inputs)
    model = tf.keras.Model(inputs, out)
    return model


model_1 = create_model(inputs[0])
model_2 = create_model(inputs)

# Works
model_1_2 = tf.keras.models.Model.from_config(model_1.get_config())
# Does not work
model_2_2 = tf.keras.models.Model.from_config(model_2.get_config())

from collections import OrderedDict
import tensorflow as tf


batch_size = 3
input_values = (
    OrderedDict((
        ('a', tf.random.uniform((batch_size, 1))),
        ('b', tf.random.uniform((batch_size, 4))),
    )),
    (
        tf.random.uniform((batch_size, 3)),
        tf.random.uniform((batch_size, 4)),
    ),
)

inputs = tf.nest.map_structure(
    lambda x: tf.keras.layers.Input(tf.shape(x)[1:]), input_values)


def create_model(inputs):
    out = tf.concat(tf.nest.flatten(inputs), axis=-1)
    model = tf.keras.Model(inputs, out)
    return model


model = create_model(inputs)
model_2 = tf.keras.models.Model.from_config(model.get_config())

from collections import OrderedDict

import tensorflow as tf


batch_size = 3
input_values = (
    OrderedDict((
        ('a', tf.random.uniform((batch_size, 1))),
        ('b', tf.random.uniform((batch_size, 4))),
    )),
    (
        tf.random.uniform((batch_size, 3)),
        tf.random.uniform((batch_size, 4)),
    ),
)

inputs = tf.nest.map_structure(
    lambda x: tf.keras.layers.Input(tf.shape(x)[1:]), input_values)


sequential_model = tf.keras.Sequential((
    tf.keras.layers.Lambda(
        lambda x: tf.concat(tf.nest.flatten(x), axis=-1)),
))

outputs = sequential_model(inputs)

print(outputs)