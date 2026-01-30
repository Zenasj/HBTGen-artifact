from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras

assert tf.__version__ == '2.3.0'

initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        keras.layers.Conv2D(32, 5, strides=2, activation="relu"),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.Conv2D(32, 3, activation="relu"),
    ]
)


feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    #Note the error!
    blah=[layer.output for layer in initial_model.layers], 
)