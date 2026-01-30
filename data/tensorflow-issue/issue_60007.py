import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (Activation, BatchNormalization,Dense, Flatten, Conv2D ,Input,
                                     Concatenate, Conv2D, Dense, Dropout,
                                     GlobalAveragePooling2D, Lambda,
                                     MaxPooling2D, add)
from tensorflow.keras.models import Model
img = np.random.random((1,112,112,3))

def create_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(128, 1, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
    ])

  return model


input_shape     = [112, 112, 3]
inputs = Input(shape=input_shape)
model = create_model()

for i in range(5):
    print(model.predict(img)[0][0:10])
# or run up core ------------------------

@tf.function
def test_step():
    return model(img,training = False)
for i in range(5):
    print(test_step()[0][0:10])