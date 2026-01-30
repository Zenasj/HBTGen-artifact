from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

tf.profiler.experimental.start('logdir_path')

inputs = tf.ones((10, 100, 100, 4))
conv = tf.keras.layers.Conv2D(
    10, (3, 3), strides=(1, 1), padding='valid',)
outputs = conv(inputs)
_ = outputs.numpy()

tf.profiler.experimental.stop()

import tensorflow as tf

tensorboard_dir = 'logdir_path'
training_hooks = []
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)
training_hooks.append(tensorboard_callback)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='sgd', loss='mse')


x = tf.ones((100, 4))
y = tf.ones((100,))

# This builds the model for the first time:
model.fit(x, y, batch_size=32, epochs=10, callbacks=training_hooks)