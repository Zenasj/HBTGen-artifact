import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.d = tf.keras.layers.Dense(2)

    # "dunder"-methods (__...__) typically are called implicitly by Python
    def __call__(self, x, training=True, mask=None):
        return self.d(x)

...

model(tf.random.normal((1, 3))) # no explicit .call here