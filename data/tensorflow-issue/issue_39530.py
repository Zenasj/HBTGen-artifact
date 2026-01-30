from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
pretrained_resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
inputs = tf.keras.Input(shape=(256,256,1))
x = tf.keras.layers.ZeroPadding2D()(inputs)
x = tf.keras.layers.Conv2D(filters=64,
                           kernel_size=(7,7),
                           strides=(2,2),
                           padding='same')(x)
outputs = pretrained_resnet.layers[3](x)
test = tf.keras.Model(inputs, pretrained_resnet.output)