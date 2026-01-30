from tensorflow import keras
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
from typing import Union, List
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers
from tftools import TFTools


class TestServe():
    def __init__(self, tfrecords: Union[List[tf.train.Example], tf.train.Example], batch_size: int = 10, input_shape: tuple = (64, 23)) -> None:
        self.tfrecords = tfrecords
        self.batch_size = batch_size
        self.input_shape = input_shape

    def get_model(self):
        ins = layers.Input(shape=(64, 23))

        l = layers.Reshape((*self.input_shape, 1))(ins)
        l = layers.Conv2D(8, (30, 23), padding='same', activation='relu')(l)
        l = layers.MaxPool2D((4, 5), strides=(4, 5))(l)
        l = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(l)
        l = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(l)
        l = layers.MaxPool2D((2, 2), strides=(2, 2))(l)
        l = layers.Flatten()(l)

        out = layers.Dense(1, activation='softmax')(l)
        return tf.keras.models.Model(ins, out)

    def train(self):

        # Create Dataset
        dataset = TFTools.create_dataset(self.tfrecords)
        dataset = dataset.repeat(6).batch(self.batch_size)

        val_iterator = dataset.take(300).make_one_shot_iterator()
        train_iterator = dataset.skip(300).make_one_shot_iterator()

        model = self.get_model()
        model.summary()
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_iterator, validation_data=val_iterator,
                  epochs=10, verbose=1, steps_per_epoch=20)

    def predict(self, X: np.array) -> np.array:
        pass

ts = TestServe(['./ok.tfrecord', './nok.tfrecord'])
ts.train()