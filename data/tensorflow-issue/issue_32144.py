import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Layer


class MyLayer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def call(self, input, training=None, mask=None):
        self.add_metric(tf.ones([32]) * 2.0, name='two', aggregation='mean')
        return input


class MyModel(Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self._sampler = MyLayer(name='sampler')

    def call(self, input, training=None, mask=None):
        z = self._sampler(input)
        self.add_metric(tf.ones([32]) * 1.0, name='one', aggregation='mean')
        self.add_metric(tf.ones([32]) * 3.0, name='three', aggregation='mean')
        return z


def train(dataset_train, epochs):
    tf.config.experimental_run_functions_eagerly(True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    loss = tf.losses.mean_squared_error

    model = MyModel()
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)

    print('Training...')
    history = model.fit(dataset_train, epochs=epochs, verbose=1, callbacks=[])


def main():
    print('Preparing data...')
    batch_size = 32
    num_examples = 32
    xdata = np.random.uniform(size=[num_examples, 16]).astype(np.float32)
    dataset_train = tf.data.Dataset.from_tensor_slices((xdata, xdata))
    dataset_train = dataset_train.batch(batch_size, drop_remainder=True)

    train(dataset_train, epochs=3)


if __name__ == '__main__':
    main()