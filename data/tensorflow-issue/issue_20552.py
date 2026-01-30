import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_keras_model():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(300, activation='relu', input_shape=[28*28]))
  model.add(tf.keras.layers.Dense(100, activation='relu'))