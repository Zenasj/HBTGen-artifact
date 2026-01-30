import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

with strategy.scope():
  #Create model, eg.:
  #model=tf.keras.models.Sequential([ .. 
  model.compile(...)
  model.fit(...)

import tensorflow as tf
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_host(resolver.master())
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
  base = InceptionV3(weights='imagenet', input_shape=(200,150,3), include_top=False)
  model = Sequential()
  model.add(base)
  model.add(Flatten())
  model.add(Dense(128))#, activation='relu'))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(128))#, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Activation('relu'))
  model.add(Dense(100))#, activation='relu'))
  model.add(Activation('relu'))
  model.add(Dense(12))#, activation='sigmoid'))
  model.add(Activation('sigmoid'))

  model.summary()
  model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])
  model.fit(trainX, trainy, epochs=5, batch_size=32, validation_split=0.08)

import tensorflow as tf
import numpy as np
import os

tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
batchSize = 64
numImages = 1000 * batchSize
width = 50
height = 50

print("Img", str(numImages))

randomImages = np.random.rand(numImages,width,height,3).astype('float32') 
randomLabels = np.random.randint(2, size=numImages).astype('int32') 

#model
resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_address)
tf.contrib.distribute.initialize_tpu_system(resolver)
strategy = tf.contrib.distribute.TPUStrategy(resolver)

with strategy.scope():
  inputs = tf.keras.layers.Input(shape=(width, height, 3))
  outputLayer = inputs
  for i in range(0,3):
    outputLayer = tf.keras.layers.Conv2D(filters=64, kernel_size = 3, padding = 'valid')(outputLayer)
    outputLayer = tf.keras.layers.BatchNormalization()(outputLayer)
    outputLayer = tf.keras.layers.ReLU()(outputLayer)
  outputLayer = tf.keras.layers.Flatten()(outputLayer)
  outputLayer = tf.keras.layers.Dense(2)(outputLayer)
  outputLayer = tf.keras.layers.BatchNormalization()(outputLayer)
  outputLayer = tf.keras.layers.Softmax()(outputLayer)

  model = tf.keras.Model(inputs = inputs, outputs = outputLayer)
  
  model.compile(
      optimizer=tf.train.AdamOptimizer(learning_rate = 0.0001),
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      metrics=['sparse_categorical_accuracy']
    )

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_host(resolver.master())
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
    model = get_model(le.classes_, 4)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(clipnorm=1.), metrics=['accuracy'])
batch_size=128
h = model.fit(X_train, [y_train[:,0], y_train[:,1], y_train[:,2], y_train[:,3]],
                validation_data=(X_test, [y_test[:,0], y_test[:,1], y_test[:,2], y_test[:,3]]),
                steps_per_epoch=len(y_train)//batch_size-1,
                validation_steps=len(y_test)//batch_size-1,
                verbose=1,
                epochs=100,
                batch_size=batch_size,
                callbacks=[
                    keras.callbacks.TerminateOnNaN(),
                    keras.callbacks.EarlyStopping(patience=5, verbose=1),
                    keras.callbacks.ReduceLROnPlateau(factor=0.9, patience=0, verbose=1, min_lr=1e-6),
                    keras.callbacks.ModelCheckpoint('model', period=0),
                ])