from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

tensorflow.python.keras.layers.normalization_v2.BatchNormalization

tensorflow.python.keras.layers.normalization.BatchNormalization

import tensorflow as tf

model = tf.keras.Sequential([
	tf.keras.layers.BatchNormalization()
])
model.build(input_shape=(1,))
model.save('/tmp/model.h5')

loaded_model = tf.keras.models.load_model('/tmp/model.h5')

# True
print(isinstance(model.layers[0], tf.keras.layers.BatchNormalization))

# False
print(isinstance(loaded_model.layers[0], tf.keras.layers.BatchNormalization))

# AttributeError: module 'tensorflow' has no attribute 'python'
import tensorflow.python.keras.layers.normalization
print(isinstance(loaded_model.layers[0], tensorflow.python.keras.layers.normalization.BatchNormalization))

for l in model.layers:
    if isinstance(l, tf.keras.layers.BatchNormalization):
        l.training = False

import tensorflow as tf

model = tf.keras.applications.resnet50.ResNet50()
assert(isinstance(model.layers[3], tf.keras.layers.BatchNormalization)), f"Layer is {model.layers[3]}, but expected {tf.keras.layers.BatchNormalization}."