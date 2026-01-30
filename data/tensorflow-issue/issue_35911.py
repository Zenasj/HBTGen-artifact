from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np


class ZerosFirstEpochOnesAfter(tf.keras.utils.Sequence):
    def __init__(self):
        self.is_epoch_0 = True

    def __len__(self):
        return 2

    def on_epoch_end(self):
        print('on_epoch_end')
        self.is_epoch_0 = False

    def __getitem__(self, item):
        if self.is_epoch_0:
            print("First epoch")
            return np.zeros((16, 1)), np.zeros((16,))
        else:
            return np.ones((16, 1)), np.ones((16,))


if __name__ == '__main__':
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_dim=1, activation="softmax"))

    model.compile(
        optimizer='Adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(ZerosFirstEpochOnesAfter(), epochs=5,)