from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras

#### BAD (raise AttributeError):
my_softplus = keras.layers.Activation("softplus")
#my_softplus = keras.layers.Lambda(tf.nn.softplus)
#my_softplus = keras.layers.Lambda(lambda X: tf.nn.softplus(X))
#my_softplus = keras.layers.Lambda(tf.function(lambda X: tf.nn.softplus(X)))

#### GOOD:
#my_softplus = "softplus"
#my_softplus = tf.nn.softplus
#my_softplus = lambda X: tf.nn.softplus(X)
#my_softplus = tf.function(lambda X: tf.nn.softplus(X))

model = keras.models.Sequential([
    keras.layers.Dense(1, activation=my_softplus, input_shape=[5])
])

from tensorflow import keras

class MyActivation(keras.layers.Layer):
    def __init__(self, activation=None, **kwargs):
        self.activation = keras.layers.Activation(activation)
        super(MyActivation, self).__init__(**kwargs)
    def call(self, X):
        return self.activation(X + 1.)

model = keras.models.Sequential([
    MyActivation(activation="softplus", input_shape=[5])
])