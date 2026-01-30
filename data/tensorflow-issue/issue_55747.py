import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import tensorflow.python as tf_python
import os
from datetime import datetime
from absl import app, flags
import math
import time
import sys

batch_size = 32


policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
tf.keras.mixed_precision.experimental.set_policy(policy)

tf.config.optimizer.set_jit(True)
os.environ['XLA_FLAGS'] = "--xla_dump_to='./xla_logs'"

H = 32
W = 32
C = 64
input_shape = (batch_size, H, W, C)

class EinSumLayer(tf.keras.layers.Layer):
    def __init__(self, name='', **kwargs):
        super(EinSumLayer, self).__init__(**kwargs)
        self._name = name

    def call(self, inputs):
        return tf.einsum('ij, jk->ik', inputs[0], inputs[1])


class TrainingModel(tf.keras.Model):
    def __init__(self):
        super(TrainingModel, self).__init__()

        # x
        # 1x1 conv 64, BN 64, Relu
        # 3x3 conv 64, BN 64, Relu
        # 1x1 conv 256, BN 256
        # Residual add : output + x
        #
        self.conv1 = tf.keras.layers.Conv2D(filters=16, input_shape=input_shape[1:],
                                            kernel_size=(1, 1), strides=(1, 1),
                                            padding='same', data_format='channels_last',
                                            activation=None, use_bias=True)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                            padding='same', data_format='channels_last',
                                            activation=None, use_bias=True)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                                            padding='same', data_format='channels_last',
                                            activation=None, use_bias=True)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()

        self.add = tf.keras.layers.Add()

        # Pooling
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same',
                                               data_format='channels_last')
        self.relu4 = tf.keras.layers.ReLU()

        # Dense layer pattern
        self.flatten = tf.keras.layers.Flatten(data_format='channels_last')

        # Activation none is supposed to check for MatMul + BiasAdd fusion
        # With activation BiasAdd gets fused with Relu
        # It is weird that smaller Dense layer sizes do not result in XLA fusion
        self.dense1 = tf.keras.layers.Dense(64, activation=None, use_bias=True)
        self.dense3 = tf.keras.layers.Dense(32, activation=None, use_bias=True)
        self.dense_out = tf.keras.layers.Dense(1, activation='relu', use_bias=True)

        self.ein_sum = EinSumLayer('ein_sum')
        self.relu5 = tf.keras.layers.ReLU()

        self.optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def call(self, x):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.add([x, y])

        x = self.pool1(x)
        x = self.relu4(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense3(x)
        x = self.ein_sum((x, x))
        x = self.relu5(x)
        x = self.dense_out(x)

        return x


model = TrainingModel()
steps = 20
input = tf.random.uniform(shape=[batch_size * steps, H, W, C], dtype=tf.float32)
target = tf.random.uniform(shape=[batch_size * steps, 1], dtype=tf.float32)
train_dataset = tf.data.Dataset.from_tensor_slices((input, target))
train_dataset = train_dataset.batch(batch_size)

# model(input)

def train_step(model_xy, x, y):
    with tf.GradientTape() as tape:
        pred = model_xy(x, training=True)
        y = tf.cast(y, dtype=tf.float16)
        loss = tf.math.reduce_mean(pred - y)

    grads = tape.gradient(loss, model_xy.trainable_weights)
    model_xy.optimizer.apply_gradients(zip(grads, model_xy.trainable_weights))

for epochs in range(1):
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(model, x_batch_train, y_batch_train)

model.compile(optimizer='adam', loss='mse')
model.fit(train_dataset)