from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

BATCH_SIZE = 2

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.outputs = tf.Variable(tf.zeros((BATCH_SIZE, 10)),
                               trainable=False)
    self.layer = tf.keras.layers.Dense(10)

  def call(self, inputs):
    return self.layer(inputs)

  def train_step(self, data):
    x, y = data
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      self.outputs.assign(y_pred)
      loss = self.compiled_loss(
          y, y_pred, regularization_losses=self.losses)
    trainable_variables = self.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}

class MyCallback(tf.keras.callbacks.Callback):
  def on_train_batch_end(self, batch, logs=None):
    print(tf.reduce_sum(self.model.outputs))

model = MyModel()
model.compile('sgd', 'mse')
x, y = np.ones((10, 100)), 10 * np.ones((10, 10))
model.fit(x, y, batch_size=BATCH_SIZE, callbacks=[MyCallback()], verbose=2)