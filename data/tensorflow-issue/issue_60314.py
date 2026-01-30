from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import keras

x = tf.constant([[0.1,0.90]])
y = tf.constant([[0.0,.0]])

# First approach 
with tf.GradientTape() as g:
  g.watch(x)
  z = keras.Sequential(
         keras.layers.Dense(2, activation = "softmax", use_bias = False)
      )(x)
  z = tf.losses.categorical_crossentropy(y, z)
dz_dx = g.gradient(z, x)
print(dz_dx)



x = tf.constant([[0.1,0.90]])
y = tf.constant([[0.0,.0]])

# Second approach 
with tf.GradientTape() as g:
  g.watch(x)
  z = keras.Sequential(
         [keras.layers.Dense(2, activation = "linear", use_bias = False),   
          keras.layers.Softmax(axis = -1)]
      )(x)
  z = tf.losses.categorical_crossentropy(y, z)
dz_dx = g.gradient(z, x)
print(dz_dx)

Step1 
import tensorflow as tf
import keras
tf.keras.utils.set_random_seed(42)
x = tf.constant([[0.1,0.90]])
y = tf.constant([[0.0,.0]])
z = tf.keras.Sequential(
         [keras.layers.Dense(2,activation = "linear", use_bias = False),
          keras.layers.Softmax()]
      )(x)
print(z)
z = tf.losses.categorical_crossentropy(y, z)
print(z)