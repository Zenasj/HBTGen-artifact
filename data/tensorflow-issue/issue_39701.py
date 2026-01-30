from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class Model(tf.keras.Model):
  def __init__(self,):
    super(Model, self).__init__()
    self.layer = tf.keras.layers.Dense(100)
  def call(self, x):
    return self.layer(x)

model = Model()
model.build((1,2))
weights = model.trainable_variables

with tf.GradientTape(persistent=True) as tape:
    output = model(tf.zeros([1,2]))

gradients = tape.jacobian(output, weights, experimental_use_pfor=False)