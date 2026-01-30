import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model

print (tf.__version__)
batch_size = 10
num_input = 5
num_output = 2
positive_weight = 10000


from tensorflow.python.keras import backend as K


def weighted_cross_entropy(y_true, y_pred, class_weights=[1, positive_weight]):
    """ Sample weights/class weights are broken in TensorFlow 1.12.

    See issue: https://github.com/tensorflow/tensorflow/issues/23767
    """
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    return -K.sum(y_true * K.log(y_pred) * K.variable(class_weights), -1)

def _to_one_hot(y, num_classes):
    """ Compute one-hot encoding of an array. """
    y_one_hot = np.zeros((len(y), num_classes)).astype(np.float32)
    y_one_hot[np.arange(len(y)), y.squeeze()] = 1

    return y_one_hot


class DummyGenerator(object):

    def class_weights(self):
        np.random.seed(7)
        while True:
            X = np.random.rand(batch_size, num_input)
            y = np.random.randint(num_output, size=batch_size)
            y = _to_one_hot(y, 2)
            yield X, y

    def sample_weights(self):
        np.random.seed(7)
        while True:
            X = np.random.rand(batch_size, num_input)
            y = np.random.randint(num_output, size=batch_size)
            w = y * (positive_weight - 1) + 1
            y = _to_one_hot(y, 2)
            yield X, y, w


def dummy_model():
    inputs = Input(shape=(num_input,))
    outputs = Dense(2, activation="softmax")(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model


model = dummy_model()
model.save_weights("~/fixed.hdf5")
gen = DummyGenerator()

print("Training with sample_weights")
model.load_weights("~/fixed.hdf5")
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.fit_generator(gen.sample_weights(), steps_per_epoch=1000, epochs=1)

print("Training with class_weights")
model.load_weights("~/fixed.hdf5")
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.fit_generator(gen.class_weights(), steps_per_epoch=1000, epochs=1, class_weight={0:1, 1:positive_weight})

print("Training with loss_weights")
model.compile(loss=weighted_cross_entropy, optimizer="adam")
model.load_weights("~/fixed.hdf5")
model.fit_generator(gen.class_weights(), steps_per_epoch=1000, epochs=1)