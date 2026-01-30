from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


def my_summary(x):
  tf.summary.scalar("mean_functional", tf.reduce_mean(x))
  return x

class MyLayer(tf.keras.layers.Layer):
    def __init(self):
        super(MyLayer, self).__init__()

    def call(self, inputs):
        tf.summary.scalar("mean_subclass", tf.reduce_mean(inputs))
        return inputs

inputs = tf.keras.Input(10)
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Lambda(my_summary)(x)
outputs = MyLayer()(outputs)
model = tf.keras.Model(inputs, outputs)
model.compile('sgd', 'mse')

# Make sure to set `update_freq=N` to log a batch-level summary every N batches.
# In addition to any `tf.summary` contained in `Model.call`, metrics added in
# `Model.compile` will be logged every N batches.
tb_callback = tf.keras.callbacks.TensorBoard('./logs240', update_freq=1)

inputs_data = tf.ones([16, 10])
labels_data = tf.ones([16, 10])

dataset = (
    tf.data.Dataset.from_tensors(inputs_data)
    .map(
        lambda x: (
            inputs_data,
            labels_data,
            None,
        )
    ).repeat(100)
)
model.fit(dataset, callbacks=[tb_callback], epochs=5)