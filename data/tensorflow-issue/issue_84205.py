import random
from tensorflow import keras
from tensorflow.keras import layers

import os
import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.random.set_seed(42)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=tf.float32)
inputs = [x]



class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=1, kernel_size=4, padding='valid', activation='relu')

    def call(self, x):
        x = tf.reshape(x, [1, 3, 3, 1])
        x = self.conv(x)
        return x


model = Model()
model(*inputs)
print("succeed on eager")



class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=1, kernel_size=4, padding='valid', activation='relu')

    @tf.function(jit_compile=True)
    def call(self, x):
        x = tf.reshape(x, [1, 3, 3, 1])
        x = self.conv(x)
        return x


model = Model()
model(*inputs)
print("succeed on XLA")