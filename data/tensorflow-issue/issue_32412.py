from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
import os

print(tf.__version__)


class ConvBn2D(tf.keras.Model):
    def __init__(self, c_out, kernel_size=3):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters=c_out, kernel_size=kernel_size,
                                           strides=1, padding="SAME",
                                           use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-7)

    def call(self, inputs):
        res = tf.nn.relu(self.bn(self.conv(inputs)))
        return res


class FNet(tf.keras.Model):
    def __init__(self, start_kernels=64, weight=0.125, **kwargs):
        super().__init__(**kwargs)
        c = start_kernels
        self.max_pool = tf.keras.layers.MaxPooling2D()
        self.init_conv_bn = ConvBn2D(c, kernel_size=3)
        self.c0 = ConvBn2D(c, kernel_size=3)

        self.c1 = ConvBn2D(c * 2, kernel_size=3)
        self.c2 = ConvBn2D(c * 2, kernel_size=3)

        self.c3 = ConvBn2D(c * 2, kernel_size=3)
        self.c4 = ConvBn2D(c * 2, kernel_size=3)

        self.pool = tf.keras.layers.GlobalMaxPool2D()
        self.linear = tf.keras.layers.Dense(10, use_bias=False)
        self.weight = weight

    def call(self, x):
        h = self.max_pool(self.c0(self.init_conv_bn(x)))
        h = self.max_pool(self.c2(self.c1(h)))
        h = self.max_pool(self.c4(self.c3(h)))
        h = self.pool(h)
        h = self.linear(h) * self.weight
        return h


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train = train.map(lambda x,y:(tf.cast(x,tf.float32),tf.cast(y,tf.int64))).map(lambda x,y:(x/255.0,y)).batch(512)

# model = FNet(start_kernels=8)

model = FNet(start_kernels=8,dynamic=True)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              loss=loss)

callbacks=[]
model.fit(train, epochs=2, callbacks=callbacks,verbose=1)