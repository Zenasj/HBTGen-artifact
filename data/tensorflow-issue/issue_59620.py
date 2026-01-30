from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

def get_model():
    mnist_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu', use_bias=False),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Dense(256, activation='relu', use_bias=False),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Dense(10, use_bias=False)])
    return mnist_model


def get_model_weights(model):
    weights = []
    for index, layer in enumerate(model.layers):
        layer_weights = model.get_layer(layer.name).weights    
        layer_weights.numpy()
        weights.append(layer_weights)
    return weights

get_model_weights(get_model())

for weightTensors in layer_weights:
      weights.append(weightTensors.numpy())