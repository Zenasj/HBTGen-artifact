from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

class lstm(tf.keras.Model):
    def __init__(self):
        super(lstm, self).__init__()

        self.embedding = tf.keras.layers.Embedding(300, 2, mask_zero=True, trainable=True)
        self.encoder = tf.keras.layers.LSTM(2, return_sequences=True, return_state=False)

    def call(self, inputs):
        output = self.embedding(inputs)
        output = self.encoder(output)
        return output[0]

input_questions = np.array([[5, 12, 13, 189, 10, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(float)
output = np.array([[-0.00299482, 0.00096033]])

dataset = tf.data.Dataset.from_tensor_slices((input_questions, output)).shuffle(2).batch(1)

model = lstm()
model.compile(tf.keras.optimizers.Adadelta(1.0), tf.keras.losses.MeanSquaredError())
model.fit(dataset, epochs=10, verbose=2)
for sample, target in dataset.take(1):
    model(sample)

import tensorflow as tf
import numpy as np

class lstm(tf.keras.Model):
    def __init__(self):
        super(lstm, self).__init__()

        self.masking = tf.keras.layers.Masking()
        self.embedding = tf.keras.layers.Embedding(300, 2, trainable=True)
        self.encoder = tf.keras.layers.LSTM(2, "sigmoid", return_sequences=True, return_state=False)

    def call(self, inputs):
        output = self.masking(inputs)
        output = self.embedding(output)
        output = self.encoder(output)
        return output[0]

input_questions = np.array([[5, 12, 13, 189, 10, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(float)
output = np.array([[-0.00299482, 0.00096033]])

dataset = tf.data.Dataset.from_tensor_slices((input_questions, output)).shuffle(2).batch(1)

model = lstm()
model.compile(tf.keras.optimizers.Adadelta(1.0), tf.keras.losses.MeanSquaredError())
model.fit(dataset, epochs=10, verbose=2)
for sample, target in dataset.take(1):
    model(sample)

import tensorflow as tf
import numpy as np

class lstm(tf.keras.Model):
    def __init__(self):
        super(lstm, self).__init__()

        self.masking = tf.keras.layers.Masking()
        self.embedding = tf.keras.layers.Embedding(300, 2, trainable=True)
        self.encoder = tf.keras.layers.LSTM(2, "tanh", return_sequences=True, return_state=False)

    def call(self, inputs):
        output = self.masking(inputs)
        output = self.embedding(output)
        output = self.encoder(output)
        return output[0]

input_questions = np.array([[5, 12, 13, 189, 10, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(float)
output = np.array([[-0.00299482, 0.00096033]])

dataset = tf.data.Dataset.from_tensor_slices((input_questions, output)).shuffle(2).batch(1)

model = lstm()
model.compile(tf.keras.optimizers.Adadelta(1.0), tf.keras.losses.MeanSquaredError())
model.fit(dataset, epochs=10, verbose=2)
for sample, target in dataset.take(1):
    model(sample)