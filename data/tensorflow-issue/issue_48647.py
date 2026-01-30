from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([
keras.layers.Softmax(axis=0, input_shape=(3, 3))])
x = tf.constant([[[0.8055, 0.0083, 0.4057], [0.1249, 0.9762, 0.5402], [0.0637, 0.1539, 0.0282]]])
print (model.predict(x,steps=1))