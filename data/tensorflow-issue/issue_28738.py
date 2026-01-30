import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])
model.compile(loss="mae", optimizer="adam")

def generator():
    i=0
    while 1:
        yield (np.array([i]),[i])
        i+=1
valData = (np.arange(10), np.arange(10))

history = model.fit_generator(generator(), steps_per_epoch=5, verbose=0, validation_data=valData)