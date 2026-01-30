from tensorflow.keras import models

from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

print(tf.__version__)

dataset_path = 'D:\\FXData\\data.csv'
checkpoint_model_json_path = 'modelBackup/model.json'
checkpoint_weights_h5_path = 'modelBackup/weights00009000.h5'
resume_from_checkpoint = True

print('reading dataset...')
column_names = ['paircode','x1o','x1h','x1l','x1c','x1v','x2o','x2h','x2l','x2c','x2v','x3o','x3h','x3l','x3c','x3v','x4o','x4h','x4l','x4c','x4v','x5o','x5h','x5l','x5c','x5v','x6o','x6h','x6l','x6c','x6v','x7o','x7h','x7l','x7c','x7v','x8o','x8h','x8l','x8c','x8v','x9o','x9h','x9l','x9c','x9v','x10o','x10h','x10l','x10c','x10v','x11o','x11h','x11l','x11c','x11v','x12o','x12h','x12l','x12c','x12v','x13o','x13h','x13l','x13c','x13v','x14o','x14h','x14l','x14c','x14v','x15o','x15h','x15l','x15c','x15v','x16o','x16h','x16l','x16c','x16v','x17o','x17h','x17l','x17c','x17v','x18o','x18h','x18l','x18c','x18v','x19o','x19h','x19l','x19c','x19v','x20o','x20h','x20l','x20c','x20v','x21o','x21h','x21l','x21c','x21v','x22o','x22h','x22l','x22c','x22v','x23o','x23h','x23l','x23c','x23v','x24o','x24h','x24l','x24c','x24v','x25o','x25h','x25l','x25c','x25v','x26o','x26h','x26l','x26c','x26v','x27o','x27h','x27l','x27c','x27v','x28o','x28h','x28l','x28c','x28v','x29o','x29h','x29l','x29c','x29v','x30o','x30h','x30l','x30c','x30v','x31o','x31h','x31l','x31c','x31v','x32o','x32h','x32l','x32c','x32v','x33o','x33h','x33l','x33c','x33v','x34o','x34h','x34l','x34c','x34v','x35o','x35h','x35l','x35c','x35v','x36o','x36h','x36l','x36c','x36v','x37o','x37h','x37l','x37c','x37v','x38o','x38h','x38l','x38c','x38v','x39o','x39h','x39l','x39c','x39v','x40o','x40h','x40l','x40c','x40v','x41o','x41h','x41l','x41c','x41v','x42o','x42h','x42l','x42c','x42v','x43o','x43h','x43l','x43c','x43v','x44o','x44h','x44l','x44c','x44v','x45o','x45h','x45l','x45c','x45v','x46o','x46h','x46l','x46c','x46v','x47o','x47h','x47l','x47c','x47v','x48o','x48h','x48l','x48c','x48v','x49o','x49h','x49l','x49c','x49v','x50o','x50h','x50l','x50c','x50v','nextclose']
dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True, skiprows = [0])

print('printing dataset tail...')
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop('nextclose')
test_labels = test_dataset.pop('nextclose')

def norm(x):
  return x
#  return (x - train_stats['mean']) / train_stats['std']

print('normalizing dataset...')  
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  print('building the model')
  model = keras.Sequential([
    layers.Dense(512, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(512, activation=tf.nn.relu), layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(1, activation='linear')
  ])
  
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
  return model

def load_model_():
  print('loading the model')
  loaded_model = load_model(checkpoint_weights_h5_path)
  return loaded_model


if resume_from_checkpoint:
  model = load_model_()
else:
  model = build_model()

model.summary()

print('testing 10 widthed batch...')
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

print('fitting the model...')
mc = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5', save_weights_only=False, period=100)

history = model.fit(
  normed_train_data, train_labels,
  epochs=100, validation_split = 0.2, verbose=2,
  batch_size=1000000, callbacks=[mc])

print('evaluating the model...')
loss, accuracy = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set MSE: {:5.2f} nextclose".format(loss))
print("Testing set Accuracy: {:5.2f} nextclose".format(accuracy))