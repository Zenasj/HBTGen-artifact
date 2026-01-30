import random
from tensorflow import keras

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Embedding(input_dim=1000, output_dim=64),
    layers.LSTM(128, return_sequences=True),
    layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy()])

data = np.random.randint(0, 1000, (32, 10))  # batch_size=32, seq_length=10
labels = np.random.randint(0, 10, (32, 10))

model.fit(data, labels, epochs=1, batch_size=32)

class InTopK(tf.keras.metrics.Mean):
    def __init__(self, k, name='in_top_k', **kwargs):
        super(InTopK, self).__init__(name=name, **kwargs)
        self._k = k

    def update_state(self, y_true, y_pred, sample_weight=None):
        matches = tf.nn.in_top_k(
            # flatten tensors
            tf.reshape(tf.cast(y_true, tf.int32), [-1]),
            tf.reshape(y_pred, [-1, y_pred.shape[-1]]),
            k=self._k)

        return super(InTopK, self).update_state(
            matches, sample_weight=sample_weight)