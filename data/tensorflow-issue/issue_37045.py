import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
import time

def random_gen():
    while True:
        time.sleep(0.01)
        yield (np.random.rand(1,), np.random.rand(1,))

dataset = tf.data.Dataset.from_generator(
        random_gen,
        (tf.float32, tf.float32),
        (tf.TensorShape((None,)), tf.TensorShape((None,))),
    )

train_set =   (dataset
               .shuffle(2000, reshuffle_each_iteration=False)
               .batch(32))
val_set =    (dataset
               .shuffle(2000, reshuffle_each_iteration=False) # Problem is here
               .batch(32))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(3))
model.compile(loss='mse', optimizer='adam')

model.fit(x=train_set,
          epochs= 5,
          steps_per_epoch= 10,
          validation_data= val_set,
          validation_steps= 5,
)