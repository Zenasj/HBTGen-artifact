from tensorflow.keras import layers
from tensorflow.keras import optimizers

class Swish(keras.layers.Layer):
    def __init__(self):
        super(Swish, self).__init__()
        self.weight = self.add_weight(initializer='uniform',trainable=True)

    def __call__(self, inputs):
        return inputs+tf.sigmoid(self.weight*inputs)

import os
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
import math
import tensorflow as tf
from tensorflow import keras

class Swish(keras.layers.Layer):
    def __init__(self):
        super(Swish, self).__init__()
        self.weight = self.add_weight(initializer='uniform',trainable=True)

    def __call__(self, inputs):
        return inputs+tf.sigmoid(self.weight*inputs)


class Conv(keras.Model):
    def __init__(self,filters,kernel_size=1,strides=1,padding='valid'):
        super(Conv, self).__init__()
        self.conv = keras.layers.Conv2D(filters,kernel_size,strides,padding)
        self.bn = keras.layers.BatchNormalization()
        self.ac = Swish()

    def __call__(self,inputs):
        return self.ac(self.bn(self.conv(inputs)))


class SEBlock(keras.Model):
    def __init__(self, filters):
        super(SEBlock, self).__init__()
        self.conv0 = keras.layers.Conv2D(filters//4,1,1)
        self.drop = keras.layers.Dropout(0.25)
        self.conv1 = keras.layers.Conv2D(filters,1,1)
        self.bn = keras.layers.BatchNormalization()
        self.ac = Swish()

    def __call__(self,inputs):
        x = self.conv1(self.drop(self.conv0(tf.reduce_mean(inputs,[1,2],keepdims=True))))
        return self.ac(self.bn(tf.sigmoid(x)*inputs))


class ResBlock(keras.Model):
    def __init__(self, filters):
        super(ResBlock, self).__init__()
        self.conv0 = keras.layers.Conv2D(filters//4,1,1)
        self.drop = keras.layers.Dropout(0.25)
        self.conv1 = keras.layers.Conv2D(filters,3,1,'same')
        self.bn = keras.layers.BatchNormalization()
        self.ac = Swish()

    def __call__(self,inputs):
        x = self.conv1(self.drop(self.conv0(inputs)))
        return self.ac(self.bn(inputs+x))

def mcn_520(width, growth,input_shape=[256,256,3]):
    fs = int(width*growth)
    inputs=keras.layers.Input(input_shape)
    x=keras.layers.Conv2D(fs,8,2)(inputs)
    x=keras.layers.MaxPool2D(2)(x)
    x1=Conv(fs//width)(SEBlock(fs)(x))
    x2=Conv(fs//width)(ResBlock(fs)(x))
    for i, depth in enumerate([2, 3, 5, 4]):
        for _ in range(int(6*depth)):
            fs+=int(math.sqrt(fs*width))
            t=keras.layers.Concatenate()([x,x1,x2])
            t=keras.layers.Dropout(0.25)(t)
            t=Conv(fs//width, 1, 1)(t)
            t=keras.layers.Dropout(0.25)(t)
            x1=SEBlock(fs//width)(t)
            x2=ResBlock(fs//width)(t)
            t=keras.layers.Concatenate()([t,x1,x2])
            t=keras.layers.Dropout(0.25)(t)
            t=Conv(growth,1,1)(t)
            x=keras.layers.Concatenate()([x,t])
        if i != 3:
            fs //= 2
            x=keras.layers.MaxPool2D(2)(Conv(fs)(x))
            x1=keras.layers.MaxPool2D(2)(Conv(fs//width)(x1))
            x2=keras.layers.MaxPool2D(2)(Conv(fs//width)(x2))
    x=keras.layers.GlobalMaxPool2D()(x)
    x=keras.layers.Dropout(0.25)(x)
    outputs=keras.layers.Dense(1000,activation='softmax')(x)
    return keras.Model(inputs=inputs,outputs=outputs,name='MCN520')

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')

starategy=tf.distribute.MirroredStrategy()
with starategy.scope():
    model=MCN.mcn_520(2,24)
    model.summary()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(),
        metrics=[tf.keras.metrics.TopKCategoricalAccuracy(1,'top1'),tf.keras.metrics.TopKCategoricalAccuracy(5,'top5')]
        )
    fit_ds,val_ds=ds(BATCH_SIZE)
    model.fit(
        fit_ds, # please use your database
        epochs=1000000,
        steps_per_epoch=BATCHS_PER_APLY_GRADIENTS*200,
    )

import os
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
import math
import tensorflow as tf
from tensorflow import keras

class Swish(keras.layers.Layer):
    def __init__(self):
        super(Swish, self).__init__()
        self.weight = self.add_weight(initializer='uniform',trainable=True)

    def __call__(self, inputs):
        return inputs+tf.sigmoid(self.weight*inputs)


class Conv(keras.Model):
    def __init__(self,filters,kernel_size=1,strides=1,padding='valid'):
        super(Conv, self).__init__()
        self.conv = keras.layers.Conv2D(filters,kernel_size,strides,padding)
        self.bn = keras.layers.BatchNormalization()
        self.ac = Swish()

    def __call__(self,inputs):
        return self.ac(self.bn(self.conv(inputs)))


class SEBlock(keras.Model):
    def __init__(self, filters):
        super(SEBlock, self).__init__()
        self.conv0 = keras.layers.Conv2D(filters//4,1,1)
        self.drop = keras.layers.Dropout(0.25)
        self.conv1 = keras.layers.Conv2D(filters,1,1)
        self.bn = keras.layers.BatchNormalization()
        self.ac = Swish()

    def __call__(self,inputs):
        x = self.conv1(self.drop(self.conv0(tf.reduce_mean(inputs,[1,2],keepdims=True))))
        return self.ac(self.bn(tf.sigmoid(x)*inputs))


class ResBlock(keras.Model):
    def __init__(self, filters):
        super(ResBlock, self).__init__()
        self.conv0 = keras.layers.Conv2D(filters//4,1,1)
        self.drop = keras.layers.Dropout(0.25)
        self.conv1 = keras.layers.Conv2D(filters,3,1,'same')
        self.bn = keras.layers.BatchNormalization()
        self.ac = Swish()

    def __call__(self,inputs):
        x = self.conv1(self.drop(self.conv0(inputs)))
        return self.ac(self.bn(inputs+x))

def mcn_520(width, growth,input_shape=[256,256,3]):
    fs = int(width*growth)
    inputs=keras.layers.Input(input_shape)
    x=keras.layers.Conv2D(fs,8,2)(inputs)
    x=keras.layers.MaxPool2D(2)(x)
    x1=Conv(fs//width)(SEBlock(fs)(x))
    x2=Conv(fs//width)(ResBlock(fs)(x))
    for i, depth in enumerate([2, 3, 5, 4]):
        for _ in range(int(6*depth)):
            fs+=int(math.sqrt(fs*width))
            t=keras.layers.Concatenate()([x,x1,x2])
            t=keras.layers.Dropout(0.25)(t)
            t=Conv(fs//width, 1, 1)(t)
            t=keras.layers.Dropout(0.25)(t)
            x1=SEBlock(fs//width)(t)
            x2=ResBlock(fs//width)(t)
            t=keras.layers.Concatenate()([t,x1,x2])
            t=keras.layers.Dropout(0.25)(t)
            t=Conv(growth,1,1)(t)
            x=keras.layers.Concatenate()([x,t])
        if i != 3:
            fs //= 2
            x=keras.layers.MaxPool2D(2)(Conv(fs)(x))
            x1=keras.layers.MaxPool2D(2)(Conv(fs//width)(x1))
            x2=keras.layers.MaxPool2D(2)(Conv(fs//width)(x2))
    x=keras.layers.GlobalMaxPool2D()(x)
    x=keras.layers.Dropout(0.25)(x)
    outputs=keras.layers.Dense(1000,activation='softmax')(x)
    return keras.Model(inputs=inputs,outputs=outputs,name='MCN520')

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')

starategy=tf.distribute.MirroredStrategy()
with starategy.scope():
    model=mcn_520(2,24)
    model.summary()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(),
        metrics=[tf.keras.metrics.TopKCategoricalAccuracy(1,'top1'),tf.keras.metrics.TopKCategoricalAccuracy(5,'top5')]
        )