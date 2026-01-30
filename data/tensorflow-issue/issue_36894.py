import random
from tensorflow import keras
from tensorflow.keras import layers

class GRU4REC(tf.keras.Model):
    def __init__(self, vocabulary_size, embedding_dim, hidden_units):
        super(GRU4REC, self).__init__()
        self.Embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_dim, mask_zero=True)
        self.FC1 = tf.keras.layers.Dense(hidden_units * 3, tf.nn.relu)
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.RELU = tf.keras.layers.ReLU()
        self.GRU = tf.keras.layers.GRU(hidden_units, return_sequences=True)
        self.FC2 = tf.keras.layers.Dense(vocabulary_size)
        self.BN2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.Embedding(inputs)
        x = self.FC1(x)
        x = self.BN1(x)
        print(f'before relu: {x._keras_mask}') # print a Tensor
        x = self.RELU(x)
        print(f'after relu: {x._keras_mask}') # print None
        x = self.GRU(x)
        x = self.FC2(x)
        x = self.BN2(x)
        x = self.RELU(x)
        # pred = tf.argmax(x, -1)
        # tf.print('++++++++++')
        # tf.print(tf.shape(pred))
        # eq = tf.equal(inputs, tf.cast(pred, tf.int32))
        # tf.print(tf.reduce_sum(tf.cast(eq, tf.float32)))
        # tf.print('---------')
        return x

import numpy as np
import tensorflow as tf


class GRU4REC(tf.keras.Model):
    def __init__(self, vocabulary_size, embedding_dim, hidden_units):
        super(GRU4REC, self).__init__()
        self.Embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_dim, mask_zero=True)
        self.FC1 = tf.keras.layers.Dense(hidden_units * 3, tf.nn.relu)
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.RELU = tf.keras.layers.ReLU()
        self.GRU = tf.keras.layers.GRU(hidden_units, return_sequences=True)
        self.FC2 = tf.keras.layers.Dense(vocabulary_size)
        self.BN2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.Embedding(inputs)
        x = self.FC1(x)
        x = self.BN1(x)
        print(f'before relu: {x._keras_mask}')  # print a Tensor
        x = self.RELU(x)
        print(f'after relu: {x._keras_mask}')  # print None
        x = self.GRU(x)
        x = self.FC2(x)
        x = self.BN2(x)
        x = self.RELU(x)
        return x


X_input = tf.keras.layers.Input([None])
encoder = GRU4REC(20000, 128, 100)
model = tf.keras.Model(inputs=X_input, outputs=encoder(X_input))

# After compile, GRU4REC.call() will be invoked and the Tensor `x` before and after ReLU will be printed to the console.
model.compile(tf.optimizers.Adam(), tf.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.metrics.SparseCategoricalAccuracy()])


def random_sequence_generator():
    for i in range(1000):
        sequence = np.random.randint(0, 19999, np.random.randint(3, 20, 1))
        yield sequence[:-1], sequence[1:]


train_dataset = tf.data.Dataset.from_generator(random_sequence_generator, output_types=(tf.int32, tf.int32)).shuffle(
    1000).padded_batch(32, ((None,), (None,)))

valid_dataset = tf.data.Dataset.from_generator(random_sequence_generator,
                                               output_types=(tf.int32, tf.int32)).padded_batch(32, ((None,), (None,)))

model.evaluate(valid_dataset)
model.fit(train_dataset, epochs=1, steps_per_epoch=10)
model.evaluate(valid_dataset)