from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

class MyDense(tf.keras.layers.Layer):
    def __init__(self, units):
        super(MyDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(units=self.units)

    def call(self, inputs, mask=None):
        return self.dense(inputs)

inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
model = tf.keras.models.Model(inputs=inputs, outputs=MyDense(units=1)(inputs))

checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory='./', max_to_keep=None)  # no checkpoint present
status = checkpoint.restore(manager.latest_checkpoint)

inputs_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 1])
called_model = model(inputs_placeholder)

with tf.Session() as session:
    status.initialize_or_restore(session)
    session.run(called_model, feed_dict={inputs_placeholder: np.array([[1.]])})