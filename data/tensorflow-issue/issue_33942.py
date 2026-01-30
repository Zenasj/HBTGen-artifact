from tensorflow.keras import layers
from tensorflow.keras import models

ipt = Input(batch_shape=batch_shape)
x   = Conv2D(6, (8, 8), strides=(2, 2), activation='relu')(ipt)
x   = Flatten()(x)
out = Dense(6, activation='softmax')(x)

ipt = Input(batch_shape=batch_shape)
x   = Conv2D(6, (8, 8), strides=(2, 2), activation='relu')(ipt)
x   = Conv2D(6, (8, 8), strides=(2, 2), activation='relu')(x)
x   = Flatten()(x)
out = Dense(6, activation='softmax')(x)

one_epoch_loss = [1.6814, 1.6018, 1.6577, 1.6789, 1.6878, 1.7022, 1.6689]
one_epoch_acc  = [0.2630, 0.3213, 0.2991, 0.3185, 0.2583, 0.2463, 0.2815]

batch_shape = (32, 64, 64, 3)
num_samples = 1152

ipt = Input(batch_shape=batch_shape)
x   = Conv2D(6, (8, 8), strides=(2, 2), activation='relu')(ipt)
x   = Conv2D(6, (8, 8), strides=(2, 2), activation='relu')(x)
x   = Flatten()(x)
out = Dense(6, activation='softmax')(x)
model = Model(ipt, out)
model.compile('adam', 'sparse_categorical_crossentropy')

X = np.random.randn(num_samples, *batch_shape[1:])
y = np.random.randint(0, 6, (num_samples, 1))

reset_seeds()
model.fit(x_train, y_train, epochs=5, shuffle=False)

import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
np.random.seed(1)
import random
random.seed(2)

import tensorflow as tf
session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
sess = tf.Session(config=session_conf) # single-threading; TF1-only

def reset_seeds():
    np.random.seed(1)
    random.seed(2)
    if tf.__version__[0] == '2':
        tf.random.set_seed(3)
    else:
        tf.set_random_seed(3)
    print("RANDOM SEEDS RESET")
reset_seeds()

from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model
import keras.backend as K

K.set_floatx('float64')