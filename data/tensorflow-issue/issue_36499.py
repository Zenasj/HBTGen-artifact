import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

X = tf.keras.Input((1,))
X2 = tf.keras.layers.Lambda(lambda x: tf.fill((tf.shape(X)[0],1),2.5))(X)
model = tf.keras.Model(inputs=X,outputs=X2)
model.predict(np.random.randn(10,1))

X2 = tf.keras.layers.Lambda(lambda x: tf.random.uniform(tf.shape(x)))(X) 
X2 = tf.keras.layers.Lambda(lambda x: tf.zeros_like(x))(X)
X2 = tf.keras.layers.Lambda(lambda x: tf.ones_like(x)*3.5)(X)

import tensorflow as tf
import numpy as np

X = tf.keras.Input((1,))
X2 = tf.keras.layers.Lambda(lambda x: tf.fill((tf.shape(X)[0],1),2.5))(X)
model = tf.keras.Model(inputs=X,outputs=X2)
model.predict(np.random.randn(10,1))

X2 = tf.keras.layers.Lambda(lambda x: tf.fill((tf.shape(X)[0],1),2.5))(X)

X2 = tf.keras.layers.Lambda(lambda x: tf.fill((tf.shape(x)[0],1),2.5))(X)