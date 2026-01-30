from tensorflow import keras
from tensorflow.keras import layers

import os
import numpy as np
import tensorflow as tf
print(tf.__version__)
import efficientnet.keras as efn 

batch_size = 32 
num_classes = 10
epochs = 3
official_efficient_net = False

## Model 
input_shape = (32,32,3)
if official_efficient_net:
    base_model  = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,
                                                                    weights="imagenet", 
                                                                    input_shape=input_shape)
else:
    base_model = efn.EfficientNetB0(include_top=False,
                                        weights="imagenet", 
                                        input_shape=input_shape)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
dense_layer = tf.keras.layers.Dense(10, activation='softmax')
Model  = tf.keras.Sequential([base_model, global_average_layer, dense_layer])

## Training Data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

## Training
Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn = Model.fit(x_train, y_train, batch_size=batch_size, 
                epochs=epochs, validation_data=(x_test,y_test))

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255