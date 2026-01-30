import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


inputs = tf.random.uniform((4, 3))

sequential_model = tf.keras.Sequential((
    tf.keras.layers.Lambda(lambda x: tf.concat(tf.nest.flatten(x), axis=-1)),
))

_ = sequential_model(inputs)

sequential_model_2 = tf.keras.Model.from_config(sequential_model.get_config())