import random
from tensorflow.keras import layers

#!/usr/bin/env python3
import argparse
import glob
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras


def read_dataset(filename, columns, field_defaults, input_size, output_size, stride, input_features):
    def decode_csv(row):
        fields = tf.decode_csv(row, record_defaults=field_defaults, field_delim=',')
        all_columns = dict(zip(columns, fields))
        return all_columns

    def split_window(window):
        inputs = tf.reshape(tf.concat(window['value'][0:input_size], axis=1), [input_size, input_features])
        outputs = tf.reshape(tf.concat(window['value'][input_size:input_size + output_size], axis=1),
                             [output_size, input_features])

        return inputs, outputs

    dataset = tf.data.TextLineDataset(filenames=filename)
    dataset = dataset.map(decode_csv)
    dataset = dataset.apply(tf.contrib.data.sliding_window_batch(window_size=input_size + output_size, stride=stride))
    dataset = dataset.map(split_window)
    dataset = dataset.repeat()

    return dataset


if __name__ == "__main__":
    COLUMNS = ['value']
    FIELD_DEFAULTS = [[0.0]]
    INPUT_FEATURES = 1

    epochs = 1
    steps = 1
    INPUT_SIZE = 4
    OUTPUT_SIZE = 2
    STRIDE = 1
    input_train = "./data/"

    input_train_list = glob.glob(input_train + "*")

    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(OUTPUT_SIZE, activation=None))
    set = read_dataset(input_train_list, COLUMNS, FIELD_DEFAULTS, INPUT_SIZE, OUTPUT_SIZE, STRIDE, INPUT_FEATURES)

    model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='mse', metrics=['mse'])
    model.fit(set, epochs=epochs, steps_per_epoch=steps)

0.047910000000000785
3.0999999999892225e-05
0.0160979999999995
2.9000000000500847e-05
0.01716599999999957
2.800000000036107e-05
2.9999999999752447e-05
0.019235000000000113

import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='mse', metrics=['mae'])

data = np.random.random((1000, 32)).astype(np.float32)
labels = np.random.random((1000, 10)).astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30)

vae.fit(X_train, batch_size=32, epochs=100) # wrong
vae.fit(X_train, X_train, batch_size=32, epochs=100) # hooray!