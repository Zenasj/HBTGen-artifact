import random
from tensorflow import keras
from tensorflow.keras import layers

# _*_ coding:utf-8 _*_
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
layers = tf.keras.layers
time_func = lambda: time.clock()*1000

class CRNNEncoder(tf.keras.Model):
    def __init__(self, configs, name=None):
        super(CRNNEncoder, self).__init__(name=name)
        self.vocab_size = configs['vocab_size']
        self.conv1 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')
        self.conv2 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')
        self.conv3 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')
        self.conv4 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')
        self.padd4 = layers.ZeroPadding2D(padding=(0, 1))
        self.pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')
        self.conv5 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')
        self.bncv5 = layers.BatchNormalization(axis=-1, name='bnconv5')
        self.conv6 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')
        self.bncv6 = layers.BatchNormalization(axis=-1, name='bnconv6')
        self.pddd6 = layers.ZeroPadding2D(padding=(0, 1))
        self.pool6 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')
        self.conv7 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='valid', name='conv7')
        self.final_layer = tf.keras.layers.Dense(self.vocab_size, name='ctc_decoder_linear')

    def get_feature_step(self, widths):
        return tf.cast((tf.cast(widths, tf.float32)/4.0), dtype=tf.int32)
    @tf.function
    def call(self, inputs, widths, training=True):
        tf.print('call input:', inputs.shape)
        features = self.conv1(inputs)
        features = self.pool1(features)
        features = self.conv2(features)
        features = self.pool2(features)
        features = self.conv3(features)
        features = self.conv4(features)
        features = self.padd4(features)
        features = self.pool4(features)
        features = self.conv5(features)
        features = self.bncv5(features, training=training)
        features = self.conv6(features)
        features = self.bncv6(features, training=training)
        features = self.pddd6(features)
        features = self.pool6(features)
        features = self.conv7(features)
        cnn_features = tf.reduce_max(features, axis=1)
        # rnn_features = self.run_bilstm1(rnn_features)
        rnn_features = cnn_features
        final_logits = self.final_layer(rnn_features)
        widths = self.get_feature_step(widths)
        return cnn_features, rnn_features, widths, final_logits

def run_forward_noloss():
    batch = 24 
    imgh = 48
    imgw = 1024 
    imgc = 3
    vocab_size = 1424
    configs = {'vocab_size': vocab_size}
    model = CRNNEncoder(configs)
    images = tf.random.uniform([batch, imgh, imgw, imgc], minval=-1, maxval=1)
    widths = tf.fill([batch], imgw)
    logit_mean = tf.keras.metrics.Mean('logit_mean')
    init_time = time_func()
    for i in range(200):
        cnn_features, rnn_features, widths, final_logits = model(images, widths)
        final_logits = tf.reduce_mean(final_logits)
        logit_mean.update_state(final_logits)
    fini_time = time_func()
    print('time:', fini_time-init_time)
    print(logit_mean.result().numpy)

def run_forward_withloss():
    batch = 24
    imgh = 48
    imgw = 1024
    imgc = 3
    txtlen = 64
    vocab_size = 1424
    eos_id = vocab_size - 1
    configs = {'vocab_size': vocab_size}
    model = CRNNEncoder(configs)
    images = tf.random.uniform([batch, imgh, imgw, imgc], minval=-1, maxval=1)
    widths = tf.fill([batch], imgw)
    labels = tf.fill([batch, txtlen], 0)
    labels_len = tf.fill([batch], txtlen)
    ctc_loss_mean = tf.keras.metrics.Mean('ctc_loss_mean')
    init_time = time_func()
    for i in range(200):
        cnn_features, rnn_features, widths, final_logits = model(images, widths)
        ctc_loss = tf.nn.ctc_loss(labels=labels,
                                  logits=final_logits,
                                  label_length=labels_len,
                                  logit_length=widths,
                                  blank_index=eos_id,
                                  logits_time_major=False)

        ctc_loss = tf.reduce_mean(ctc_loss)
        ctc_loss_mean.update_state(ctc_loss)
    fini_time = time_func()
    print('time:', fini_time - init_time)
    print(ctc_loss_mean.result().numpy)

if __name__ == '__main__':
    run_forward_noloss()
    run_forward_withloss()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# _*_ coding:utf-8 _*_
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
layers = tf.keras.layers
time_func = lambda: time.clock()*1000

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class CRNNEncoder(tf.keras.Model):
    def __init__(self, configs, name=None):
        super(CRNNEncoder, self).__init__(name=name)

        self.vocab_size = configs['vocab_size']
        self.conv1 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')
        self.conv2 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')
        self.conv3 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')
        self.conv4 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')

        self.padd4 = layers.ZeroPadding2D(padding=(0, 1))
        self.pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')
        self.conv5 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')
        self.bncv5 = layers.BatchNormalization(axis=-1, name='bnconv5')
        self.conv6 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')
        self.bncv6 = layers.BatchNormalization(axis=-1, name='bnconv6')
        self.pddd6 = layers.ZeroPadding2D(padding=(0, 1))
        self.pool6 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')
        self.conv7 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='valid', name='conv7')
        self.final_layer = tf.keras.layers.Dense(self.vocab_size, name='ctc_decoder_linear')

    def get_feature_step(self, widths):
        return tf.cast((tf.cast(widths, tf.float32)/4.0), dtype=tf.int32)

    @tf.function
    def call(self, inputs, widths, training=True):
        tf.print('call input:', inputs.shape)
        features = self.conv1(inputs)
        features = self.pool1(features)
        features = self.conv2(features)
        features = self.pool2(features)
        features = self.conv3(features)
        features = self.conv4(features)
        features = self.padd4(features)
        features = self.pool4(features)
        features = self.conv5(features)
        features = self.bncv5(features, training=training)
        features = self.conv6(features)
        features = self.bncv6(features, training=training)
        features = self.pddd6(features)
        features = self.pool6(features)
        features = self.conv7(features)
        cnn_features = tf.reduce_max(features, axis=1)
        # rnn_features = self.run_bilstm1(rnn_features)
        rnn_features = cnn_features
        final_logits = self.final_layer(rnn_features)
        widths = self.get_feature_step(widths)
        return cnn_features, rnn_features, widths, final_logits


def run_forward_noloss():
    batch = 24 
    imgh = 48
    imgw = 1024 
    imgc = 3
    vocab_size = 1424
    configs = {'vocab_size': vocab_size}
    model = CRNNEncoder(configs)
    images = tf.random.uniform([batch, imgh, imgw, imgc], minval=-1, maxval=1)
    widths = tf.fill([batch], imgw)
    logit_mean = tf.keras.metrics.Mean('logit_mean')
    init_time = time_func()
    for i in range(2000):
        cnn_features, rnn_features, widths, final_logits = model(images, widths)
        final_logits = tf.reduce_mean(final_logits)
        logit_mean.update_state(final_logits)

    fini_time = time_func()
    print('time:', fini_time-init_time)
    print(logit_mean.result().numpy)


def run_forward_withloss():
    batch = 24
    imgh = 48
    imgw = 1024
    imgc = 3
    txtlen = 64
    vocab_size = 1424
    eos_id = vocab_size - 1
    configs = {'vocab_size': vocab_size}
    model = CRNNEncoder(configs)
    images = tf.random.uniform([batch, imgh, imgw, imgc], minval=-1, maxval=1)
    widths = tf.fill([batch], imgw)
    labels = tf.fill([batch, txtlen], 0)
    labels_len = tf.fill([batch], txtlen)
    ctc_loss_mean = tf.keras.metrics.Mean('ctc_loss_mean')

    @tf.function
    def run_step(images_, widths_, labels_, labels_len_):
        cnn_features, rnn_features, widths, final_logits = model(images_, widths_)

        ctc_loss = tf.nn.ctc_loss(labels=labels_,
                                  logits=final_logits,
                                  label_length=labels_len_,
                                  logit_length=widths,
                                  blank_index=eos_id,
                                  logits_time_major=False)

        ctc_loss = tf.reduce_mean(ctc_loss)
        ctc_loss_mean.update_state(ctc_loss)


    init_time = time_func()
    for i in range(2000):
        run_step(images, widths, labels, labels_len)

    fini_time = time_func()
    print('time:', fini_time - init_time)
    print(ctc_loss_mean.result().numpy)


if __name__ == '__main__':
    #run_forward_noloss()
    run_forward_withloss()