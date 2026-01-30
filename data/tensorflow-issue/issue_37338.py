from tensorflow import keras
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras import layers
class Model():
    def __init__(self):
        self.build_model()

    def build_model(self):
        input1 = layers.Input(shape=5)

        net = tf.keras.models.Sequential([
            layers.Dense(2),
            layers.Dense(1),
        ])

        out1 = net(input1)
        """
        out1 = layers.Dense(2)(input1)
        out1 = layers.Dense(1)(out1)
        """

        self.model = tf.keras.Model(inputs=input1, outputs=out1)

model = Model()
s1 = "exp\\model"
model.model.save(s1)
model2 = tf.keras.models.load_model(s1)

# Correct if not using tf.keras.Sequential
out1 = layers.Dense(2)(input1)
out1 = layers.Dense(1)(out1)