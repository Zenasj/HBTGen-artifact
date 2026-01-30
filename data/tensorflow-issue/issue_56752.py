import random
from tensorflow import keras

# conding=utf8

from tensorflow.keras import Input, layers, optimizers, losses, metrics
from tensorflow.keras import mixed_precision
import tensorflow as tf

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def makeModel():

    input_1 = Input(shape=(1))

    dense1 = layers.Dense(256, activation='leaky_relu')(input_1)
    dense2 = layers.Dense(256, activation='leaky_relu')(dense1)
    dense3 = layers.Dense(256, activation='leaky_relu')(dense2)
    dense4 = layers.Dense(256, activation='leaky_relu')(dense3)

    outputs = layers.Dense(1)(dense4)

    model = tf.keras.Model(input_1, outputs)
    model.summary()

    model.compile(optimizer=optimizers.SGD(learning_rate=0.01),
                  loss=losses.MeanAbsoluteError(), jit_compile=True)

    return model



def _parse_func(serialized):
    return tf.cast(serialized / tf.int64.max, tf.float32), tf.random.uniform([1])

BATCH_SIZE = 24

dataset = tf.data.Dataset.random()
dataset = dataset.take(1024000)
dataset = dataset.map(_parse_func, tf.data.AUTOTUNE, False)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with strategy.scope():
    model = makeModel()

model.fit(dataset, epochs=100)