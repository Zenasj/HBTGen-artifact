import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf


class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__(name='')

        self.l1 = tf.keras.layers.Conv2D(10, 3)
        self.o1 = tf.keras.layers.Conv2D(2, 1)
        self.o2 = tf.keras.layers.Conv2D(3, 1, name='o2')

        # print(self._output_names)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        y = self.l1(inputs)
        y1 = self.o1(y)
        y1 = tf.reshape(y1, [batch_size, -1, 2])
        y2 = self.o2(y)
        y2 = tf.reshape(y2, [batch_size, -1, 3])

        return y1, y2


class Loss1(tf.keras.losses.Loss):

    def call(self, targets, predictions):
        losses = tf.math.abs(predictions - targets)
        return tf.reduce_sum(losses, axis=2)


x = tf.random.uniform([10, 16, 16, 3])
y1 = tf.random.uniform([10, 14*14, 2])
y2 = tf.random.uniform([10, 14*14, 3])
w1 = tf.random.uniform([10, 14*14, 1])
w2 = tf.random.uniform([10, 14*14, 1])


data = tf.data.Dataset.from_tensor_slices((x, (y1, y2), {'output_1': w1, 'output_2': w2})).batch(5).repeat()


model = MyModel()
optimizer = tf.keras.optimizers.SGD(0.001)

model.compile(optimizer=optimizer,
              loss=[Loss1(), Loss1()],
              loss_weights=[1., 1.],
              sample_weight_mode="temporal",
              run_eagerly=True)


model.fit(data, epochs=5, steps_per_epoch=1)

inputs = keras.Input(shape=(16, 16, 3))
y = keras.layers.Conv2D(10, 3)(inputs)
y1 = keras.layers.Conv2D(2, 1)(y)
y1 = keras.layers.Reshape((-1, 2), name='o1')(y1)
y2 = keras.layers.Conv2D(3, 1)(y)
y2 = keras.layers.Reshape((-1, 3), name='o2')(y2)

model = keras.Model(inputs=inputs, outputs=[y1, y2])