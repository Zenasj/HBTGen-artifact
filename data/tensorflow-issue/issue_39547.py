import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        if training is True:
            print("in training")
            x = self.dropout(x, training=training)
        elif training is None:
            print("training None")
        else:
            print("not in training")
            
        return self.dense2(x)

model = MyModel()

optimizer = tf.keras.optimizers.Adam(1e-4)
loss = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer, loss)
x = tf.random.normal((5,))
y = tf.ones((5,))
model.fit(x, y, epochs=1)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        if training is True:
            print("in training")
            x = self.dropout(x, training=training)
        elif training is None:
            print("training None")
        else:
            print("not in training")
            
        return self.dense2(x)

model = MyModel()

optimizer = tf.keras.optimizers.Adam(1e-4)
loss = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer, loss)
x = tf.random.normal((1, 5))
y = tf.ones((1, 5))
model.fit(x, y, epochs=1)