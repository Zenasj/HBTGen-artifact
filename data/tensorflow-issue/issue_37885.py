from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

input_layer = tf.keras.layers.Input(shape=(32,32,3))
output = tf.keras.layers.BatchNormalization()(input_layer)
model = tf.keras.Model(inputs=input_layer, outputs=output)

yaml_out = model.to_yaml()
model2 = tf.keras.models.model_from_yaml(yaml_out)