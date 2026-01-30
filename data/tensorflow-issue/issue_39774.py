import random
from tensorflow import keras
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf

class ToyModel(tf.keras.Model):
  def call(self, inputs):
    # Inputs are scalar, so we should only have a batch dimesion.
    print("Input shape is {}".format(inputs.shape))
    return tf.reduce_sum(inputs)

toy_model = ToyModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss = tf.keras.losses.MeanSquaredError()

data_size = 10
data_inputs = np.float32(np.random.randn(data_size))
data_outputs = np.float32(np.random.randn(data_size))
dataset = tf.data.Dataset.from_tensor_slices(
    (data_inputs, data_outputs)).batch(data_size)

# Train iteratively
iterator = iter(dataset)
inputs, outputs = iterator.next()
with tf.GradientTape() as tape:
  predictions = toy_model(inputs)
  loss_value = loss(outputs, predictions)
grads = tape.gradient(loss_value, toy_model.trainable_weights)
optimizer.apply_gradients(zip(grads, toy_model.trainable_weights))
# Input shape is [batch_size]

# Now, try training with fit() instead
toy_model.compile(optimizer=optimizer, loss=loss)
toy_model.fit(dataset, epochs=1)  
# Input shape is [batch_size, 1]