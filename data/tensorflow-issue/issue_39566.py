import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def my_function():
    print('barfoo')
    a = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(16,))])
    print('foobar')