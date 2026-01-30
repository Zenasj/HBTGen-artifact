import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf


class WeightedSDRLoss(tf.keras.losses.Loss):

    def __init__(self, noisy_signal, reduction=tf.keras.losses.Reduction.AUTO, name='WeightedSDRLoss'):
        super().__init__(reduction=reduction, name=name)
        self.noisy_signal = noisy_signal

    def sdr_loss(self, sig_true, sig_pred):
        return (-tf.reduce_mean(sig_true * sig_pred) /
                (tf.norm(tensor=sig_pred) * tf.norm(tensor=sig_true)))

    def call(self, y_true, y_pred):
        noise_true = self.noisy_signal - y_true
        noise_pred = self.noisy_signal - y_pred
        alpha = (tf.reduce_mean(tf.square(y_true)) /
                 tf.reduce_mean(tf.square(y_true) + tf.square(self.noisy_signal - y_pred)))
        return alpha * self.sdr_loss(y_true, y_pred) + (1 - alpha) * self.sdr_loss(noise_true, noise_pred)

data_x = np.random.rand(5, 4, 1)
data_y = np.random.rand(5, 4, 1)

y_true = tf.keras.layers.Input(shape=[4, 1])
x = tf.keras.layers.Input(shape=[4, 1])
y_pred = tf.keras.layers.Activation('tanh')(x)
model = tf.keras.models.Model(inputs=[x, y_true], outputs=y_pred)
model.add_loss(WeightedSDRLoss(x)(y_true, y_pred))

train_dataset = tf.data.Dataset.from_tensor_slices(((data_x, data_y),)).batch(1)

model.compile()
model.fit(train_dataset)