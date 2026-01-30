import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

inp = tf.keras.layers.Input(shape=(84, 84, 3))
dense = tf.keras.layers.Conv2D(10, 3, activation=None)(inp)
bn = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn1')(dense)
rel = tf.keras.layers.ReLU()(bn)
flat = tf.keras.layers.Flatten()(rel)
out = tf.keras.layers.Dense(1, )(flat)
model = tf.keras.models.Model(inputs=inp, outputs=out)

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())
model.fit(x=np.random.uniform(size=(4, 84, 84, 3)), y=np.random.uniform(size=(4, 1)), epochs=1)
model.evaluate(x=np.random.uniform(size=(3, 84, 84, 3)), y=np.random.uniform(size=(3, 1)))
model.predict(x=np.random.uniform(size=(1, 84, 84, 3)))