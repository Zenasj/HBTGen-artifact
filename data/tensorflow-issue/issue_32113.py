from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)


model = MyModel()
model.build((None, 10))
model.summary()