from tensorflow.keras import layers
from tensorflow.keras import models

py
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model, load_model


class Works(Model):
    def build(self, input_shape):
        self.fc = Dense(32)

    def call(self, inputs):
        a, b = inputs
        b = self.fc(b)
        return tf.sparse.sparse_dense_matmul(a, b)


class Crashes(Model):
    def build(self, input_shape):
        self.fc = Sequential([Dense(32)])  # <<<< THIS IS THE ONLY DIFFERENCE

    def call(self, inputs):
        a, b = inputs
        b = self.fc(b)
        return tf.sparse.sparse_dense_matmul(a, b)

# Inputs
a = tf.sparse.from_dense(tf.ones((10, 10)))
b = tf.ones((10, 10))

# This works OK, no Sequential model
works = Works()
works([a, b])
save_model(works, 'works')
load_model('works')

# This crashes, it uses a Sequential model
crashes = Crashes()
crashes([a, b])
save_model(crashes, 'crashes')
load_model('crashes')  # <<<< FAILS