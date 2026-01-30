import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Lambda

inputs = Input(shape=(1,))
x = Lambda(lambda x: x*2)(inputs)
out = Dense(1)(x)
model = Model(inputs=inputs,outputs=out)

model_config = model.to_json()
tf.keras.models.model_from_json(model_config)