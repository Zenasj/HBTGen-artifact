import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf


def input_fn():
    x = np.random.random((1024, 10))
    y = np.random.randint(2, size=(1024, 1))
    x = tf.cast(x, tf.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(32)
    dataset = dataset.repeat(10)
    
    def _extract_features(_x, _y):
        features = {
            'x': _x,
            'z': tf.zeros_like(_x)
        }
        
        return features, _y

    dataset = dataset.map(_extract_features)

    return dataset


class MyModel0(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.features = tf.keras.layers.DenseFeatures([
            tf.feature_column.numeric_column('x', shape=(10,))
        ])
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=None, mask=None):
        outputs = self.features(inputs)
        outputs = self.dense1(outputs)
        outputs = self.dense2(outputs)

        return outputs


model = MyModel()
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.05))

# Works
model.run_eagerly = True
model.fit(input_fn())

# Works
model.run_eagerly = False
model.fit(input_fn())

# Works
model.run_eagerly = True
model.fit_generator(input_fn())

# Fails
model.run_eagerly = False
model.fit_generator(input_fn())