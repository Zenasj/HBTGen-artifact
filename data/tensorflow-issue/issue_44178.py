from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential([
  keras.layers.Dense(1, input_shape=(1,)),
])

model.save('my_model')