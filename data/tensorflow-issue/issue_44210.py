import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.losses import Loss
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


class BatchMeanSquaredError(Loss):

    def __init__(self, reduction='auto', name='batch_mean_squared_error'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        L = K.mean((y_pred - y_true) ** 2, axis=0)
        return L

X = np.random.random((1000, 3))
y = np.ones(shape=(1000, 3))

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(3)
    ]
)

bmse = BatchMeanSquaredError()
model.compile(loss=bmse, optimizer='sgd')

model.fit(X, y, batch_size=10, epochs=5)

tf.keras.models.save_model(model=model, filepath='model.h5')

custom_objects = {'BatchMeanSquaredError': BatchMeanSquaredError}
tf.keras.models.load_model('model.h5', custom_objects=custom_objects)