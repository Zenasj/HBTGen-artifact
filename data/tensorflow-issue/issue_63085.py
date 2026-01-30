from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_hub as hub

 # Wrap the hub layer in a Lambda layer

hub_layer_wrapper = tf.keras.layers.Lambda(lambda x: hub_layer(x))

model = tf.keras.Sequential([
    hub_layer_wrapper,  # Use the wrapped layer
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()