import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CustomLayer(tf.keras.layers.Layer):
    ...
    def call(inputs, training=True):
        outputs = custom_op(inputs, training=training)
        return outputs

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
     layer = CustomLayer()

@tf.function
def _step(input):
    return layer(input)

for iterations in range(10):
    output = strategy.run(_step, args=(inputs,)) # where inputs are an PerReplica data

class CustomLayer(tf.keras.layers.Layer):
    ...
    def call(inputs, training=True):
        outputs = custom_op(inputs, training=training)
        return outputs

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
     layer = CustomLayer()
     var = tf.Variable(initial_value=1.0, dtype=tf.float32)

@tf.function
def _step(input):
    return var * layer(input)

for iterations in range(10):
    output = strategy.run(_step, args=(inputs,)) # where inputs are an PerReplica data

class CustomLayer(tf.keras.layers.Layer):
    ...
    def call(inputs, training=True):
        outputs = custom_op(inputs, training=training)
        return outputs

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
     layer = CustomLayer()
     var = tf.Variable(initial_value=1.0, dtype=tf.float32)

@tf.function
def _step(input):
    return var * layer(input)

for iterations in range(10):
    output = strategy.run(_step, args=(inputs,)) # where inputs are an PerReplica data