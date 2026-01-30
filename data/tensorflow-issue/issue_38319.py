from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from functools import partial

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[...,None] / 255 - 1
x_test = x_test[...,None] / 255 - 1
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
model = keras.models.Sequential(
  [
    keras.layers.Conv2D(6, (3,3), activation='relu'),
    keras.layers.AveragePooling2D(),
    keras.layers.Conv2D(16, (3,3), activation='relu'),
    keras.layers.AveragePooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(units=120, activation='relu'),
    keras.layers.Dense(units=84, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax')
  ]
)

def dummy_loss(y,y2,a):
  return y+y2

model.compile(
  loss=partial(dummy_loss, a=2),
  #loss=lambda v1, v2 : dummy_loss(v1,v2,4),
  optimizer=keras.optimizers.Adam(), metrics={'output_1':'accuracy'})
model.train_on_batch(x_train[:2,...], y_train[:2,...]) # error here

def dummy_loss(a=2):
    def lossFunction(y,y2):    
        return y+y2
    return lossFunction

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn import metrics as skm
import json
from utils.training import focal_loss

class FocalLoss(tf.keras.losses.Loss):

    def __init__(self, num_class, num_sub_catg_class, global_batch_size, input_shape, reduction=tf.keras.losses.Reduction.AUTO, lambda_conf=100.0, lambda_offsets=1.0, class_weights=1.0, name='focal_loss'):
        self.global_batch_size = global_batch_size
        self.num_class = num_class
        self.num_sub_catg_class = num_sub_catg_class
        self.lambda_conf = lambda_conf
        self.lambda_offsets = lambda_offsets
        self.class_weights = class_weights
        self.num_rows, self.num_cols = input_shape[:2]
        self.name = name
        self.reduction = reduction

    def __call__(self, y_true, y_pred, sample_weight=None):
        
        num_rows, num_cols = self.num_rows, self.num_cols
        self.class_mask_tensor, self.class_sub_catg_mask_tensor = 1.0, 1.0
        if sample_weight is not None:
            print("Sample weight provided")
            sample_weight = K.cast(sample_weight, K.floatx())
            self.class_mask_tensor = sample_weight[:, :num_rows*num_cols]
            self.class_sub_catg_mask_tensor = sample_weight[:, num_rows*num_cols:2*num_rows*num_cols]

        class_y_true = y_true[:, :, :, :self.num_class]
        sub_catg_class_y_true = y_true[:, :, :, self.num_class:self.num_class+self.num_sub_catg_class]

        class_y_pred = y_pred[:, :, :, :self.num_class]
        sub_catg_class_y_pred = y_pred[:, :, :, self.num_class:self.num_class+self.num_sub_catg_class]

        class_y_true = tf.reshape(class_y_true, tf.stack([self.global_batch_size, num_rows*num_cols, self.num_class]))
        class_y_pred = tf.reshape(class_y_pred, tf.stack([self.global_batch_size, num_rows*num_cols, self.num_class]))
        class_loss = focal_loss(class_y_true, class_y_pred, mask=self.class_mask_tensor)

        sub_catg_class_y_true = tf.reshape(sub_catg_class_y_true, tf.stack([self.global_batch_size, num_rows*num_cols, self.num_sub_catg_class]))
        sub_catg_class_y_pred = tf.reshape(sub_catg_class_y_pred, tf.stack([self.global_batch_size, num_rows*num_cols, self.num_sub_catg_class]))
        sub_catg_class_loss = focal_loss(sub_catg_class_y_true, sub_catg_class_y_pred, mask=self.class_sub_catg_mask_tensor)

        mean_class_loss = tf.nn.compute_average_loss(tf.squeeze(class_loss), global_batch_size=self.global_batch_size)
        mean_sub_catg_class_loss = tf.nn.compute_average_loss(tf.squeeze(sub_catg_class_loss), global_batch_size=self.global_batch_size)

        total_loss = mean_class_loss+mean_sub_catg_class_loss
        return total_loss