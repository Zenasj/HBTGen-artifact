import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

if self.console:
    print('Tensor: {}'.format(t.name))
    print('{}\n'.format(K.get_value(t)[:10]))
else:
    tf.summary.histogram(t.name, data=t)

class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
  def _log_gradients(self, epoch):
    step = tf.cast(tf.math.floor((epoch+1)*num_instance/batch_size), dtype=tf.int64)
    writer = self._get_writer(self._train_run_name)

    with writer.as_default(), tf.GradientTape() as g:
      # here we use test data to calculate the gradients
      _x_batch = x_te[:100]
      _y_batch = y_te[:100]

      g.watch(_x_batch)
      _y_pred = self.model(_x_batch)  # forward-propagation
      loss = self.model.loss(y_true=_y_batch, y_pred=_y_pred)  # calculate loss
      gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

      # In eager mode, grads does not have name, so we get names from model.trainable_weights
      for weights, grads in zip(self.model.trainable_weights, gradients):
        tf.summary.histogram(
            weights.name.replace(':', '_')+'_grads', data=grads, step=step)
    
    writer.flush()

  def on_epoch_end(self, epoch, logs=None):  
    # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
    # but we do need to run the original on_epoch_end, so here we use the super function. 
    super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)

    if self.histogram_freq and epoch % self.histogram_freq == 0:
      self._log_gradients(epoch)

import tensorflow as tf
from tensorflow.python.keras import backend as K

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu', name='l_1st'),
  tf.keras.layers.Dense(128, activation='relu', name='l_2nd'),
  tf.keras.layers.Dense(128, activation='relu', name='l_3rd'),
  tf.keras.layers.Dense(128, activation='relu', name='l_4th'),
  tf.keras.layers.Dense(128, activation='relu', name='l_5th'),
  tf.keras.layers.Dense(128, activation='relu', name='l_6th'),
  tf.keras.layers.Dense(128, activation='relu', name='l_7th'),
  tf.keras.layers.Dense(128, activation='relu', name='l_8th'),
  tf.keras.layers.Dense(128, activation='relu', name='l_9th'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax', name='dense10')
])

l = tf.keras.losses.SparseCategoricalCrossentropy()
opt = tf.keras.optimizers.Adam(0.001)

model.compile(optimizer=opt, loss=l, metrics=['accuracy'])

class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):

  def _log_gradients(self, epoch):
    step = tf.cast(epoch, dtype=tf.int64)
    writer = self._train_writer
    # writer = self._get_writer(self._train_run_name)

    with writer.as_default(), tf.GradientTape() as g:
      # here we use test data to calculate the gradients
      _x_batch = x_train[:100]
      _y_batch = y_train[:100]

      g.watch(tf.convert_to_tensor(_x_batch))
      _y_pred = self.model(_x_batch)  # forward-propagation
      loss = self.model.loss(y_true=_y_batch, y_pred=_y_pred)  # calculate loss
      gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

      # In eager mode, grads does not have name, so we get names from model.trainable_weights
      for weights, grads in zip(self.model.trainable_weights, gradients):
        tf.summary.histogram(
            weights.name.replace(':', '_')+'_grads', data=grads, step=step)

    writer.flush()

  def on_epoch_end(self, epoch, logs=None):  
  # def on_train_batch_end(self, batch, logs=None):  
    # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
    # but we do need to run the original on_epoch_end, so here we use the super function. 
    super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)
    # super(ExtendedTensorBoard, self).on_train_batch_end(batch, logs=logs)
    if self.histogram_freq and epoch % self.histogram_freq == 0:
      self._log_gradients(epoch)

ee = ExtendedTensorBoard(histogram_freq=1, write_images=True, update_freq='batch')
model.fit(x_train, y_train, epochs=10, callbacks=[ee], validation_data=(x_test, y_test), )
# model.fit(x_train, y_train, epochs=5, callbacks=[gradient_cb, tensorboard_cb])