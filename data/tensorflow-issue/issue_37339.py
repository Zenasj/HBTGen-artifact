from tensorflow import keras
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras import layers
class Model():
    def __init__(self):
        self.build_model()

    def build_model(self):
        input1 = layers.Input(shape=(5,))
        input2 = layers.Input(shape=(5,))

        out1 = layers.Dense(1)(input1)
        out2 = layers.Dense(1)(input2)
        out = out1 - out2
        out = tf.nn.sigmoid(out)

        self.model = tf.keras.Model(inputs=[input1, input2], outputs=out)

model = Model()
s1 = "exp\\model"
model.model.save(s1)
model2 = tf.keras.models.load_model(s1)

model = Model()
s1 = "exp\\model"
model.model.save(s1)
model2 = tf.keras.models.load_model(s1)

# Correct if not using this line
out = tf.nn.sigmoid(out)