from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

initial_builing = False
model_loading = True

if initial_builing:

    layer1 = tf.keras.Input((1,),)
    layer2 = tf.keras.layers.Dense(1)

    model_output = layer2(layer1)[:,:-1]

    model = tf.keras.Model(layer1, model_output)
    model.summary()
    model.save('testmodel')

if model_loading:
    model = tf.keras.models.load_model('testmodel')
    model = tf.keras.Model(model.inputs, model.layers[-1].output[:,:-1])