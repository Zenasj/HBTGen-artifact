from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class my_class(tf.keras.Model):

    def __init__(self):
        super(my_class, self).__init__()

        self.x = [0]
        print(self.x)

model = my_class()

import tensorflow as tf

class my_class(object):

    def __init__(self):
        super(my_class, self).__init__()

        self.model = tf.keras.layers.Dense(3)
        # Specify input size
        self.model.build((32, 32, 1))
    
    def forward(self):
        image = tf.zeros((32, 32, 1))
        self.images = []
        self.images.append(image)
        self.model(self.images)

model = my_class()
model.forward()

import tensorflow as tf

tf.enable_eager_execution()

class my_class(tf.keras.Model):

    def __init__(self):
        super(my_class, self).__init__()

        self.model = tf.keras.layers.Dense(3)
        # Specify input size
        self.model.build((32, 32, 1))
    
    def forward(self):
        image = tf.zeros((32, 32, 1))
        self.images = []
        self.images.append(image)
        self.model(self.images)

model = my_class()
model.forward()