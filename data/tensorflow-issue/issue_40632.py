import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam

print(tf.__version__)

class MultiInputLinear(tf.keras.layers.Layer):
    def __init__(self, output_dim=32, n_inputs=2):
        super(MultiInputLinear, self).__init__()
        self.output_dim = output_dim
        self.n_inputs = n_inputs


    def build(self, input_shapes):
        self.input_dim = input_shapes[0][1]

        self.W = tf.concat(
            [
                self.add_weight(
                    name=f'W_{i}',
                    shape=(self.input_dim, self.output_dim),
                    initializer='random_normal',
                    trainable=True
                ) for i in range(self.n_inputs)
            ], axis=0
        )

    def call(self, inputs):  
        supports = tf.concat(inputs, axis=-1)        
        return tf.matmul(supports, self.W)

N = 100
A = [np.random.normal(size=(N, N)) for _ in range(2)]
y = np.random.binomial(1, .1, size=(N, 32))

A_in = [tf.keras.layers.Input(batch_size=N, shape=(N, )) for _ in range(2)]
Y = MultiInputLinear(y.shape[1], 2)(A_in)

model = tf.keras.models.Model(inputs=A_in, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam())

model.fit(A, y, batch_size=N)

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam

class MultiInputLinear(tf.keras.layers.Layer):
    def __init__(self, output_dim=32, n_inputs=2):
        super(MultiInputLinear, self).__init__()
        self.output_dim = output_dim
        self.n_inputs = n_inputs


    def build(self, input_shapes):
        self.input_dim = input_shapes[0][1]

        self.W_list = [
                self.add_weight(
                    name=f'W_{i}',
                    shape=(self.input_dim, self.output_dim),
                    initializer='random_normal',
                    trainable=True
                ) for i in range(self.n_inputs)
            ]

    def call(self, inputs):  
        supports = tf.concat(inputs, axis=-1)
        W = tf.concat(self.W_list, axis=0)

        return tf.matmul(supports, W)

N = 100
A = [np.random.normal(size=(N, N)) for _ in range(2)]
y = np.random.binomial(1, .1, size=(N, 32))

A_in = [tf.keras.layers.Input(batch_size=N, shape=(N, )) for _ in range(2)]
Y = MultiInputLinear(y.shape[1], 2)(A_in)

model = tf.keras.models.Model(inputs=A_in, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam())

model.fit(A, y, batch_size=N)