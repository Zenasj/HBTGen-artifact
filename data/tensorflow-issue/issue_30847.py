from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import os
from tensorflow.compat.v1 import enable_eager_execution
from tensorflow.data import Dataset
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

enable_eager_execution()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

strategy = MirroredStrategy()
n = strategy.num_replicas_in_sync
print('Number of replicas: {}'.format(n))

dataset = Dataset.from_tensors(({'input_1': ['This is a string'],
                                 'input_2': [1., 2., 3., 2.]},
                                {'output': [3.]})).repeat(256).shuffle(8).batch(2 * n)

with strategy.scope():
    input_1 = Input(shape=(1,), dtype='string', name='input_1')
    input_2 = Input(shape=(4,), name='input_2')
    output = Dense(1, name='output')(input_2)
    model = Model([input_1, input_2], [output], name='my_model')
    model.compile(optimizer='adam', metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')

model.fit(dataset, epochs=2)
model.evaluate(dataset)

import tensorflow as tf
print('Tensorflow version:', tf.__version__)

strategy = tf.distribute.MirroredStrategy()
n = strategy.num_replicas_in_sync
print('Number of replicas: {}'.format(n))

dataset = tf.data.Dataset.from_tensors(({#'input_1': ['This is a string'],
                                 'input_2': [1., 2., 3., 2.]},
                                {'output': [3.]})).repeat(256).shuffle(8).batch(2 * n, drop_remainder=True)

with strategy.scope():
#     input_1 = tf.keras.layers.Input(shape=(1,), dtype='string', name='input_1')
    input_2 = tf.keras.layers.Input(shape=(4,), name='input_2')
    output = tf.keras.layers.Dense(1, name='output')(input_2)
    model = tf.keras.models.Model([#input_1,
                                   input_2], [output], name='my_model')
    model.compile(optimizer='adam', metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')

model.fit(dataset, epochs=2)
model.evaluate(dataset)