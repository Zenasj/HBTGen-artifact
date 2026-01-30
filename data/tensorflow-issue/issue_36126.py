from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import tensorflow as tf

import xlrd

import pandas as pd
import csv
import numpy as np
import csv

tf.compat.v1.enable_eager_execution()

train_data_url = "https://www.dropbox.com/s/mug8rjlniftu065/train_data_csv.csv?dl=0"

test_data_url = "https://www.dropbox.com/s/std8rt6lezl79ti/test_data_csv.csv?dl=0"

train_file_path = tf.keras.utils.get_file("train_data_csv.csv", train_data_url)
test_file_path = tf.keras.utils.get_file("test_data_csv.csv", test_data_url)

np.set_printoptions(precision = 3, suppress=True)

#!head {train_file_path}

Label_Column = 'Besucher'
Labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900]

train_dataset = tf.data.experimental.make_csv_dataset(
    './Data/train_data_csv.csv',
    batch_size = 52609,
    select_columns = ['Datum','Uhrzeit','Wochentag','Wochenende','Ferien','Feiertag','Brueckentag','Schneechaos','Streik','Besucher'],
    label_name = 'Besucher',
    num_epochs = 1,
    shuffle = False)



test_dataset = tf.data.experimental.make_csv_dataset(
    './Data/alt_test_data_csv.csv',
    batch_size = 1,
    select_columns = ['Datum','Uhrzeit','Wochentag','Wochenende','Feiertag','Besucher'],
    label_name = 'Besucher',
    num_epochs = 1 ,
    shuffle = False)


def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))




show_batch(train_dataset)

def pack(features, label):
    return tf.stack(list(features.values()), axis = -1), label

packed_dataset1 = train_dataset.map(pack)
#packed_dataset2 = test_dataset.map(pack)

for features, labels in packed_dataset1.take(1):
    print(features.numpy())
    print()
    print(labels.numpy())

example_batch, labels_batch = next(iter(train_dataset))


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):

        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf. cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis = -1)
        features['numeric'] = numeric_features

        return features, labels

NUMERIC_FEATURES = ['Datum','Uhrzeit','Wochentag','Wochenende','Ferien','Feiertag','Brueckentag','Schneechaos','Streik']

packed_train_data = train_dataset.map(
    PackNumericFeatures(NUMERIC_FEATURES)
)
packed_test_data = train_dataset.map(
    PackNumericFeatures(NUMERIC_FEATURES)
)

show_batch(packed_train_data)

example_batch, labels_batch = next(iter(packed_train_data))


desc = pd.read_csv("./Data/train_data_csv.csv")[NUMERIC_FEATURES].describe()
desc

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

def normalize_numeric_data(data, mean, std):

    return(data-mean)/std

normalizer = functools.partial(normalize_numeric_data, mean = MEAN, std = STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
numeric_column

example_batch['numeric']

numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
numeric_layer(example_batch).numpy()

preprocessing_layer = numeric_layer

print(preprocessing_layer(example_batch).numpy()[0])

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

model.fit(train_data, epochs = 20)

test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

import tensorflow as tf
import tensorflow_datasets as tfds

dataset = tfds.load("mnist", split="train")
val_dataset = tfds.load("mnist", split="test")

def preprocessing(data):
    return tf.cast(data["image"], tf.float32), data["label"]

dataset = (
    dataset.cache()
    .shuffle(4 * 1024)
    .map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE,)
    .batch(1024)
    .prefetch(1)
)

val_dataset = (
    val_dataset.cache()
    .map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE,)
    .batch(1024)
    .prefetch(1)
)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", "sparse_top_k_categorical_accuracy"],
)

model.fit(dataset, epochs=5, validation_data=val_dataset)