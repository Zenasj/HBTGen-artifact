import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class NumpySequence(Sequence):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

workers = 1
steps_per_epoch, epochs = 100, 10
N, H, W, C = 4, 1000, 1000, 10

x_train = []
y_train = []
rng = np.random.default_rng()
for step in tqdm.tqdm(range(steps_per_epoch)):
    x_train.append(rng.standard_normal(size=(N, H, W, C), dtype=np.float32))
    y_train.append(rng.standard_normal(size=(N, H, W, C), dtype=np.float32))

data_sequence = NumpySequence(x_train, y_train)

model = tf.keras.models.Sequential([tf.keras.layers.Activation("linear")])
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy())
model.fit(data_sequence, epochs=epochs, workers=workers)