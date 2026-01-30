import random
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class ThreeLayerMLP(keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.dense_1 = layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = layers.Dense(64, activation='relu', name='dense_2')
        self.pred_layer = layers.Dense(10, name='predictions')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.pred_layer(x)


model = ThreeLayerMLP(name='3_layer_mlp')

x_train, y_train = (np.random.random(
    (60000, 784)), np.random.randint(10, size=(60000, 1)))
x_test, y_test = (np.random.random(
    (10000, 784)), np.random.randint(10, size=(10000, 1)))

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop())

callback = tf.keras.callbacks.TensorBoard(
    'subclass_logs',
    update_freq=2,
)
history = model.fit(x_train,
                    y_train,
                    batch_size=64,
                    epochs=10,
                    callbacks=[callback])