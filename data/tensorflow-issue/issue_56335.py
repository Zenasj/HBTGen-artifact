import math
from tensorflow import keras
from tensorflow.keras import models

import tensorflow as tf

class CustomModel(tf.keras.models.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.var = tf.Variable(initial_value=0.1, dtype=tf.float32, trainable=True)

    def loss(self):
        return tf.math.reduce_mean(tf.concat([self.var]*10,0))


model = CustomModel()

with tf.GradientTape() as gt:
    loss = model.loss()
    gt.gradient(loss, model.trainable_variables)