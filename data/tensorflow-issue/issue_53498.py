import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from models.model_attention import AttentionModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense


inputs = {'f1': tf.keras.layers.Input(name='f1', sparse=True, shape=(40, 1), dtype='float32'),
          'f2': tf.keras.layers.Input(name='f2', sparse=True, shape=(40, 1), dtype='float32')}

features = [tf.feature_column.sequence_numeric_column('f1', dtype=tf.float32),
            tf.feature_column.sequence_numeric_column('f2', dtype=tf.float32)]

input_layer, _ = tf.keras.experimental.SequenceFeatures(features)(inputs)
lstm_out = LSTM(128, return_sequences=False)(input_layer)
lstm_out = Dropout(0.2)(lstm_out)
lstm_out = Dense(1, activation='tanh')(lstm_out)
model = tf.keras.models.Model(inputs, lstm_out)
model.compile(loss='mse', metrics='mae', optimizer='Adam')


def gen():
    batch = 4
    while True:
        x1 = tf.sparse.from_dense(np.random.random((batch, 40, 1)))
        x2 = tf.sparse.from_dense(np.random.random((batch, 40, 1)))
        x = {'f1': x1, 'f2': x2}
        y = np.random.random((batch, 1))
        yield x, y


x, y = gen().__next__()
# x, y yielded from generator works
model.fit(x, y, epochs=2, verbose=2)
g = gen()
# TypeError: Input must be a SparseTensor.
model.fit(g, steps_per_epoch=2, epochs=2, verbose=2, validation_data=g, validation_steps=2)