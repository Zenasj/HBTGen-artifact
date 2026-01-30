from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.ops import state_ops

import numpy as np

class BatchCounter(tf.keras.layers.Layer):

        def __init__(self, name="batch_counter", **kwargs):
            super(BatchCounter, self).__init__(name=name, **kwargs)
            self.stateful = True
            self.batches = tf.keras.backend.variable(value=0, dtype="int32")

        def reset_states(self):
            tf.keras.backend.set_value(self.batches, 0)

        def __call__(self, y_true, y_pred):
            current_batches = self.batches * 1
            self.add_update(
              state_ops.assign_add(self.batches,
                                   tf.keras.backend.variable(value=1, dtype="int32")))
            return current_batches + 1

class DummyGenerator(object):
    """ Dummy data generator. """

    def run(self):
        while True:
            yield np.ones((10, 1)), np.zeros((10, 1))

train_gen = DummyGenerator()
val_gen = DummyGenerator()

# Dummy model
inputs = Input(shape=(1,))
outputs = Dense(1)(inputs)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss="mse", optimizer="adam", metrics=[BatchCounter()])

model.fit_generator(
    train_gen.run(),
    steps_per_epoch=5,
    epochs=200,
    validation_data=val_gen.run(),
    validation_steps=5,
    callbacks=[tf.keras.callbacks.TensorBoard()])