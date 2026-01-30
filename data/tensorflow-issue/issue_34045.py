from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
# tf.__version__ yields 2
import tensorflow_hub as hub
import numpy as np

@tf.function
def augment(x, y):
    tf.print(tf.where(tf.equal(y, 0))) # works
    # tf.print(tf.where(y == 0)) # doesn't work for some reason
    return tf.data.Dataset.from_tensors((x, y))

def gen():
    for i in range(400):
        yield np.zeros(shape=(299, 299, 3)), np.zeros(shape=(6,))

dataset = tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64), (tf.TensorShape([299,299,3]), tf.TensorShape([6])))

class Trainer:
    def __init__(self, batch_size):
        self.input_shape = [299, 299, 3]
        self.model = self.create_model()
        self.dataset = dataset
        self.dataset_size = self._get_dataset_size()
        split_point = int(0.8 * self.dataset_size)
        self.batch_size = batch_size
        self.train_ds = self.dataset.take(split_point).flat_map(augment).repeat().batch(self.batch_size)
        self.train_ds_size = split_point

        self.validation_ds = self.dataset.skip(split_point).repeat().batch(self.batch_size)
        self.validation_ds_size = self.dataset_size - split_point
        assert self.validation_ds_size + self.train_ds_size == self.dataset_size

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
                                 output_shape=[2048],
                                 input_shape=self.input_shape,
                                 trainable=False))
        model.add(tf.keras.layers.Dense(units=6, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model

    def _get_dataset_size(self):
        dataset_length = [i for i, _ in enumerate(self.dataset)][-1] + 1
        return dataset_length

    def train(self, epochs):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
            # tf.keras.callbacks.ModelCheckpoint('./model.h5', save_best_only=True),
            # tf.keras.callbacks.TensorBoard(log_dir='./logs', write_graph=True)
        ]

        self.model.fit(self.train_ds,
                       epochs=epochs,
                       steps_per_epoch=self.train_ds_size // self.batch_size,
                       validation_data=self.validation_ds,
                       validation_steps=self.validation_ds_size // self.batch_size,
                       verbose=2,
                       callbacks=callbacks)
trainer = Trainer(3)
trainer.train(epochs=2)