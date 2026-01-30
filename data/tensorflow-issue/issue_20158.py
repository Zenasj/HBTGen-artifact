import random
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import lookup_ops
from tensorflow import keras

alphabet = ['A', 'B', 'C']
table = lookup_ops.index_table_from_tensor(tf.constant(alphabet))
# generate samples of strings of different lengths
inputs = [''.join([np.random.choice(alphabet) for _ in range(5)]) for _ in range(10)]
targets = np.zeros((10, 4))

dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
def map_fn(x, y):
    x = tf.string_split([x], delimiter="").values
    x = table.lookup(x)
    x = tf.nn.embedding_lookup(tf.eye(3), x)
    return x, y
dataset = dataset.map(lambda x, y: map_fn(x, y))
dataset = dataset.repeat(100)
dataset = dataset.batch(5)

x = keras.layers.Input(shape=(5, 3), name='input')
flat = keras.layers.Flatten()(x)
y = keras.layers.Dense(4, name='dense')(flat)

model = keras.Model(x, y)
model.compile(loss='mse', optimizer='rmsprop')

model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)