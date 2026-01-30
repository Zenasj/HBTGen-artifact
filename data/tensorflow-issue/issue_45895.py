import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
inp = tf.keras.Input((1,))
out = tf.keras.layers.Dense(1)(inp)
model = tf.keras.Model(inp, out)
model.compile(loss='mse', optimizer='sgd')

x = tf.random.uniform((10, 1))
y = 2 * x + 3
dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)

sched = tf.keras.callbacks.LearningRateScheduler(lambda i: 1. / (i+1))
sched._supports_tf_logs = True  # makes ProgbarLogger display lr
# same issue with ReduceLROnPlateau
callbacks = [sched]

# add logger at end, otherwise it's inserted at front and won't print lr
callbacks.append(tf.keras.callbacks.ProgbarLogger())

model.fit(dataset, epochs=10, callbacks=callbacks)

class RecallCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
      logs['recall@1'] = evaluate_recall(...)

def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    numpy_logs = None
    for callback in self.callbacks:
      if getattr(callback, '_supports_tf_logs', False):
        callback.on_epoch_end(epoch, logs)
      else:
        if numpy_logs is None:  # Only convert once.
          numpy_logs = tf_utils.to_numpy_or_python_type(logs)
        callback.on_epoch_end(epoch, numpy_logs)