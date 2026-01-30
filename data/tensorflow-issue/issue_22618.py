import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

def estimator_fn():
    x_in = tf.keras.layers.Input(shape=[10])
    x = tf.keras.layers.Dense(16, activation='relu')(x_in)
    x_out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(x_in, x_out)
    lr = tf.Variable(0.1, trainable=False, dtype=tf.float32)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    model.compile(loss='binary_crossentropy', optimizer= optimizer)
    estimator = tf.keras.estimator.model_to_estimator(keras_model = model)
    return estimator

def input_fn():
    np.random.seed(100)
    x = np.random.random((1024, 10))
    y = np.random.randint(2, size=(1024, 1))
    x = tf.cast(x, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat()
    dataset = dataset.batch(1024)
    return dataset
  
model_estimator = estimator_fn()
model_estimator.train(input_fn=input_fn, steps=1000)

import tensorflow as tf
import numpy as np
from keras import optimizers

def estimator_fn():
    x_in = tf.keras.layers.Input(shape=[10])
    x = tf.keras.layers.Dense(16, activation='relu')(x_in)
    x_out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(x_in, x_out)
    lr = tf.Variable(0.1, trainable=False, dtype=tf.float32)
    optimizer = keras.optimizers.SGD(lr, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer= optimizer)
    estimator = tf.keras.estimator.model_to_estimator(keras_model = model)
    return estimator

def input_fn():
    np.random.seed(100)
    x = np.random.random((1024, 10))
    y = np.random.randint(2, size=(1024, 1))
    x = tf.cast(x, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat()
    dataset = dataset.batch(1024)
    return dataset
  
model_estimator = estimator_fn()
model_estimator.train(input_fn=input_fn, steps=1000)

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils import multi_gpu_model
import numpy as np

def estimator_fn():
    x_in = tf.keras.layers.Input(shape=[10])
    x = tf.keras.layers.Dense(16, activation='relu')(x_in)
    x_out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    with tf.device('/cpu:0'):
        model = tf.keras.models.Model(x_in, x_out)
    num_gpu = 2
    parallel_model = multi_gpu_model(model, gpus=num_gpu)
    
    lr = tf.Variable(0.1, trainable=True, dtype=tf.float32)
    optimizer = tf.keras.optimizers.SGD(0.01, momentum=0.0, decay=0.0, nesterov=False)
    parallel_model.model.compile(loss='binary_crossentropy', optimizer= optimizer)
    estimator = tf.keras.estimator.model_to_estimator(keras_model =  parallel_model)
    return estimator

def input_fn():
    np.random.seed(100)
    x = np.random.random((1024, 10))
    y = np.random.randint(2, size=(1024, 1))
    x = tf.cast(x, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat()
    dataset = dataset.batch(1024)
    return dataset
  
model_estimator = estimator_fn()
model_estimator.train(input_fn=input_fn, steps=1000)