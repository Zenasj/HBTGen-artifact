import random
from tensorflow import keras
from tensorflow.keras import layers

tensor = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(tensor)
tensor = tf.contrib.layers.group_norm(tensor)

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import sys
import os
import cv2
import json
from sklearn.metrics import confusion_matrix

ORIGINAL_WIDTH = 4608
ORIGINAL_HEIGHT = 4608
SCALED_WIDTH = 512
SCALED_HEIGHT = 512

tf.enable_eager_execution()

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.keras.layers.Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = tf.keras.layers.add([x, input_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):

    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = tf.keras.layers.Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    shortcut = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def get_fpn(num_channels=3):

    input = tf.keras.Input(shape=(SCALED_HEIGHT, SCALED_WIDTH, num_channels))
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(input)
    x = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    bottomup_xl = identity_block(x, 3, [64, 64, 256], stage=2, block='c') # (?, 127, 127, 256)

    x = conv_block(bottomup_xl, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    bottomup_l = identity_block(x, 3, [128, 128, 512], stage=3, block='d') # (?, 64, 64, 512)

    x = conv_block(bottomup_l, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    bottomup_m = identity_block(x, 3, [256, 256, 1024], stage=4, block='f') # (?, 32, 32, 1024)

    x = conv_block(bottomup_m, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    bottomup_s = identity_block(x, 3, [512, 512, 2048], stage=5, block='c') # (?, 16, 16, 2048)

    topdown_s = tf.keras.layers.Conv2D(1024, (1, 1), padding='same')(bottomup_s)
    x = tf.keras.layers.UpSampling2D(size=(2,2))(topdown_s)
    #x = tf.image.resize_images(topdown_s, (topdown_s.shape[1] * 2, topdown_s.shape[2] * 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #x = tf.convert_to_tensor(cv2.resize(np.array(topdown_s), dsize=None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    topdown_m = tf.keras.layers.add([x, tf.keras.layers.Conv2D(1024, (1, 1), padding='same')(bottomup_m)])

    x = tf.keras.layers.UpSampling2D(size=(2, 2))(topdown_m)
    topdown_l = tf.keras.layers.add([x, tf.keras.layers.Conv2D(1024, (1, 1), padding='same')(bottomup_l)])
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(topdown_l)
    topdown_xl = tf.keras.layers.add([x, tf.keras.layers.Conv2D(1024, (1, 1), padding='same')(bottomup_xl)])

    pyramid_s = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(topdown_s)
    pyramid_m = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(topdown_m)
    pyramid_l = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(topdown_l)
    pyramid_xl = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(topdown_xl)

    # extra_channels = num_channels - 3
    # paddings = tf.constant([[0, 0, ], [0, 0], [0, extra_channels], [0, 0]])
    # weights[0] = tf.pad(weights[0], paddings, mode='REFLECT')
    # model.set_weights(weights[:36])
    return tf.keras.Model(input, [pyramid_s, pyramid_m, pyramid_l, pyramid_xl])

def get_semantic_fpn(num_channels=3):
    def upsample(tensor, repetitions):
        print(tf.executing_eagerly())
        if repetitions == 0:
            tensor = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(tensor)
            tensor = tf.contrib.layers.group_norm(tensor) # Problematic line
            return tf.keras.layers.ReLU()(tensor)
        for _ in range(repetitions):
            tensor = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(tensor)
            tensor = tf.contrib.layers.group_norm(tensor) # Problematic line
            tensor = tf.keras.layers.ReLU()(tensor)
            tensor = tf.image.resize_images(tensor, (tensor.shape[1] * 2, tensor.shape[2] * 2))
        return tensor

    input = tf.keras.Input(shape=(SCALED_HEIGHT, SCALED_WIDTH, num_channels))
    fpn = get_fpn(num_channels)
    pyramid_s, pyramid_m, pyramid_l, pyramid_xl = fpn(input)
    from_s = upsample(pyramid_s, 3)
    from_m = upsample(pyramid_m, 2)
    from_l = upsample(pyramid_l, 1)
    from_xl = upsample(pyramid_xl, 0)

    x = tf.keras.layers.add([from_s, from_m, from_l, from_xl])
    x = tf.keras.layers.Conv2D(NUM_LABELS, (1, 1), padding='same')(x)
    output = tf.image.resize_images(x, (x.shape[1] * 4, x.shape[2] * 4))
    return tf.keras.Model(input, output)

model = get_semantic_fpn()

import tensorflow as tf
slim = tf.contrib.slim

def test():
    x = tf.placeholder(dtype=tf.float32,shape=[None,112,112,3])
    net = slim.conv2d(x,32,[3,3])
    out = tf.contrib.layers.group_norm(net,groups=4)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    import numpy as np
    inputs = np.random.rand(2,112,112,3)

    print(sess.run(out,feed_dict={x:inputs}))


if __name__ == "__main__":
    test()