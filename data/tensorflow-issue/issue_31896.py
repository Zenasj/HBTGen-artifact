import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
import os
import cv2
import tensorflow.keras.layers as K

def extract_fn(data_record):
    features = {
        'data': tf.io.FixedLenFeature([], tf.string)
    }
    sample = tf.io.parse_single_example(data_record, features)
    data = tf.image.decode_image(sample['data'])

    return data, 1.


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataset_iterator, len):
        self.dataset_iterator = dataset_iterator
        self.len = len

    def __len__(self):
        # number of batches per epoch
        return self.len

    def __getitem__(self, index):
        # Generate one batch of data

        next_element = next(self.dataset_iterator)
        x = next_element[0]
        y = next_element[1]

        return x, y


with tf.io.TFRecordWriter("dummy_dataset.tfrecords") as writer:
    data = np.float32(np.random.random(size=(1000, 1000, 3)) * 255)
    data = cv2.imencode(".png", data)[1].tostring()
    example = tf.train.Example(features=tf.train.Features(
        feature={'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))}))
    for i in range(10000):
        writer.write(example.SerializeToString())

dataset = tf.data.TFRecordDataset(["dummy_dataset.tfrecords"])
dataset = dataset.map(extract_fn)
n_batch = 3
dataset = dataset.batch(batch_size=n_batch, drop_remainder=True)

dataset = dataset.repeat(5)
dataset_iterator = iter(dataset)
next_element = next(dataset_iterator)

data_generator = DataGenerator(dataset_iterator, int(10000/n_batch))

input = K.Input(shape=(1000, 1000, 3), name='input')
net = K.Conv2D(1, 3, activation='sigmoid')(input)
output = K.GlobalAveragePooling2D()(net)
model = tf.keras.models.Model(inputs=input, outputs=output)

model.compile(loss='mse', optimizer='sgd')

model.fit(x=data_generator, epochs=10)