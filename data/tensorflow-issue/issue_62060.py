from tensorflow import keras
from tensorflow.keras import layers

import sys

import tensorflow as tf

print(f"{tf.version.VERSION=} {tf.version.GIT_VERSION=} {tf.version.COMPILER_VERSION=}")
print(f"{sys.version=}")

dividers = [0, 2, 5]

assert all(divider >= 0 for divider in dividers)
sizes = [end - start for start, end in zip(dividers[:-1], dividers[1:])]
assert all(size > 0 for size in sizes)
channels = dividers[-1]

i = tf.keras.layers.Input((channels,), name='i')
o = [
    tf.keras.layers.Lambda(lambda x: x[..., start:end],
                           name=f'slice_{start}_{end}')(i)
    for start, end in zip(dividers[:-1], dividers[1:])
]
m = tf.keras.Model(i, o, name='m')
m.build((channels,))
m.summary()
print(f"{m.input_shape=}")
print(f"{m.output_shape=}")
print(f"{m.compute_output_shape(m.input_shape)=}")
x = tf.zeros((1, channels))
print(f"{[y.shape for y in m(x)]=}")
print(f"{[y.shape for y in m.predict(x)]=}")
assert m.output_shape == m.compute_output_shape(m.input_shape)