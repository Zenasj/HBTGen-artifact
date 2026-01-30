import random
from tensorflow import keras

import tensorflow as tf
import numpy as np

inputs = tf.keras.Input(shape=(1,), dtype=tf.int64)
outputs = tf.random.normal(shape=(inputs[0, 0],))

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='sgd', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

dummy = np.zeros((10, 1))
shapes = np.random.randint(0, 11, 10).reshape(10, 1, 1)
dataset = tf.data.Dataset.from_tensor_slices(shapes)
dataset = dataset.map(lambda x: (x, tf.random.uniform(minval=0, maxval=2, shape=(x[0,0],), dtype=tf.int32)))

# eagar mode, works
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
for inputs, true in dataset:
    pred = model(inputs)
    print(true)
    print(pred)
    print(loss(true, pred))
    break

# compiled model, fails
model.fit(dataset, epochs = 1, steps_per_epoch=1)

import numpy as np

print('Tensorflow', tf.__version__)

inputs = tf.keras.Input(shape=(1,), dtype=tf.int32)
outputs = tf.random.normal(
    shape=(tf.reduce_max(inputs) - tf.reduce_min(inputs) + 1,))

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='sgd',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

def gen():
    while True:
        x = np.random.normal(5, 10, size=(5, 1)).astype('i4')
        y = np.random.randint(0, 2, size=int(np.ptp(x) + 1)).astype('i4')
        yield (x, y)

dataset = tf.data.Dataset.from_generator(
    gen, output_types=(tf.int32, tf.int32),
    output_shapes=(tf.TensorShape([5, 1]), tf.TensorShape([None])))

# eager mode, works
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
for inputs, true in dataset:
    pred = model(inputs)
    print(true.shape)
    print(pred.shape)
    print(loss(true, pred))
    break

# compiled model, fails
model.fit(dataset, epochs = 1, steps_per_epoch=1)