import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import os
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
import tensorflow as tf
from Models import MCN
from DataSets import ImageNet

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')

BATCH_SIZE=20
BATCHS_PER_APLY_GRADIENTS=1000//BATCH_SIZE

ds=ImageNet.ImageNetP()
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
        fit_ds,
        epochs=1000000,
        steps_per_epoch=BATCHS_PER_APLY_GRADIENTS*200,
        validation_data=val_ds,
        validation_steps=ds.val_images//BATCH_SIZE,
    )

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

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy import sparse

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


tf.random.set_seed(22)
np.random.seed(22) 
assert tf.__version__.startswith('2.')

batchsz = 256

# the most frequest words
total_words = 4096
max_review_len = 4995
embedding_len = 100

matrixfile = "my matrix"
targetfile = "my label"

allmatrix = sparse.load_npz(matrixfile).toarray()
target = np.loadtxt(targetfile)
print("allmatrix shape: {}；target shape: {}".format(allmatrix.shape, target.shape))

x = tf.convert_to_tensor(allmatrix, dtype=tf.float32)
y = tf.convert_to_tensor(target, dtype=tf.int32)
idx = tf.range(allmatrix.shape[0])
idx = tf.random.shuffle(idx)
x_train, y_train = tf.gather(x, idx[:int(0.7 * len(idx))]), tf.gather(y,idx[:int(0.7 * len(idx))])
x_val, y_val = tf.gather(x, idx[int(0.7 * len(idx)):int(0.8 * len(idx))]), tf.gather(y, idx[int(0.7 * len(idx)):int(0.8 * len(idx))])
x_test, y_test = tf.gather(x, idx[int(0.8 * len(idx)):]), tf.gather(y, idx[int(0.8 * len(idx)):])


db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db.shuffle(6000).batch(batchsz, drop_remainder=True).repeat()
db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = ds_val.batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)


class MyRNN(keras.Model):

    def __init__(self, units):
        super(MyRNN, self).__init__()


       
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)


        self.rnn = keras.Sequential([
            layers.GRU(units, dropout=0.5, return_sequences=True, unroll=True),
            layers.GRU(units, dropout=0.5, unroll=True)
        ])


        # fc, [b, 80, 100] => [b, 64] => [b, 1]
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        # [b, 80]
        x = inputs
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute
        # x: [b, 80, 100] => [b, 64]
        x = self.rnn(x, training=training)

        # out: [b, 64] => [b, 1]
        x = self.outlayer(x)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob

def main():
    units = 64
    epochs = 100

    import time


    t0 = time.time()
    model = MyRNN(units)
    model.compile(optimizer = keras.optimizers.Adam(0.001),
                  loss = tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit(db_train, epochs=epochs, validation_data=db_val)

    model.evaluate(db_test)


    t1 = time.time()
   
    print('total time cost:', t1-t0)


if __name__ == '__main__':
    main()

class SEBlock(keras.Model):
    def __init__(self, filters):
        super(SEBlock, self).__init__() # must have this
        self.conv0 = keras.layers.Conv2D(filters//4,1,1)
        self.drop = keras.layers.Dropout(0.25)
        self.conv1 = keras.layers.Conv2D(filters,1,1)
        self.bn = keras.layers.BatchNormalization()
        self.ac = Swish()
        
    def get_config(self):
        # I tried to use 'supper().get_config()',
        # but it seems not have this function.
        return {'filters':self.conv1.get_config()['filters']}

    def call(self,inputs):
        x = self.conv1(self.drop(self.conv0(tf.reduce_mean(inputs,[1,2],keepdims=True))))
        return self.ac(self.bn(tf.sigmoid(x)*inputs))