import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf 
import numpy as np 

train_tokens_X = np.zeros((1066673, 61, 69), dtype=np.float32)
train_tokens_X[:] = np.eye(69)[:61,:]

train_target = np.zeros((1066673, 3943), dtype=np.float32)
train_target[:,2] = 1

valid_tokens_X= np.zeros((133366, 61, 69), dtype=np.float32)
valid_tokens_X[:] = np.eye(69)[:61,:]

valid_target= np.zeros((133366, 3943), dtype=np.float32)
valid_target[:,2] = 1

output_size = 3943
max_id = 68
batch_size = 256

class Sequencer(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array(batch_x), np.array(batch_y)


train_generator = Sequencer(train_tokens_X, train_target, batch_size)
valid_generator = Sequencer(valid_tokens_X, valid_target, batch_size)

mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with mirrored_strategy.scope():
    model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=[ None, max_id+1], use_bias=False),
        keras.layers.GRU(128, return_sequences=True, use_bias=False),
        keras.layers.GRU(128, use_bias=False),
        keras.layers.Flatten(),
        keras.layers.Dense(output_size, activation="softmax")
    ])
    model.compile(loss=[focal_loss_umbertogriffo.categorical_focal_loss(alpha=.25, gamma=2)], optimizer="adam", metrics=['accuracy'])

history = model.fit(train_generator, validation_data=valid_generator, epochs=25, callbacks = callbacks, max_queue_size=10, workers=2, use_multiprocessing=True)