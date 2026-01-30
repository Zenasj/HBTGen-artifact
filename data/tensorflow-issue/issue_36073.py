from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):

  def __init__(self):

    super(MyModel, self).__init__()

    self.dense = tf.keras.layers.Dense(100, kernel_initializer="ones", trainable=False)
    self.dropout = tf.keras.layers.Dropout(0.5)


  def call(self, inputs, training=False):

    x = self.dense(inputs)

    if training:
      x = self.dropout(x, training=training)

    x = tf.reshape(tf.reduce_sum(x)/100., [1, 1]) # If we dont reshape, we get RuntimeError: Attempted to aggregate unsupported object 1.0.
    
    return x

model = MyModel()

def loss(y_true, y_pred):
    return y_pred

model.compile(optimizer="sgd", loss=loss)

x = np.ones((1, 1), dtype=np.float32)

print("Predicting :")
print(model.predict(x)) # No dropout, output is 1 as expected

print("Training :")
print(model.train_on_batch(x)) # dropout should put half of the activations of model.dense to 0, so I expect this value to be 0.5