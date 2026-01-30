from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
inp = tf.keras.Input(shape=(10,))
out = tf.keras.layers.Lambda(
        lambda x_input: x_input,
        dynamic=True,
)(inp)
model = tf.keras.Model(inputs=inp, outputs=out)

import tensorflow as tf

@tf.function
def python_control_flow_fn(tensor):
    return tf.concat([t for t in tensor],axis=0)

inp = tf.keras.Input(shape=(10,))
layer = tf.keras.layers.Lambda(python_control_flow_fn)(inp)

import tensorflow as tf
print(tf.version.GIT_VERSION, tf.version.VERSION, flush=True)
print(tf.config.list_physical_devices(), flush=True)


inp = tf.keras.Input(shape=(10,))
out = tf.compat.v1.keras.layers.Lambda(
        lambda x_input: x_input,
        dynamic=True,
)(inp)
model = tf.keras.Model(inputs=inp, outputs=out)