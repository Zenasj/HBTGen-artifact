import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_custom_lstm_model(input_shape, units=64):
       inputs = tf.keras.Input(shape=input_shape)
       x = tf.keras.layers.LSTM(units, return_sequences=True)(inputs)
       x = tf.keras.layers.LSTM(units)(x)
       outputs = tf.keras.layers.Dense(1)(x)
       return tf.keras.Model(inputs=inputs, outputs=outputs)