import random
from tensorflow import keras
from tensorflow.keras import models

feature_columns = []

categorical = tf.feature_column.categorical_column_with_vocabulary_list(
        key='f1', vocabulary_list=['x1','x2','x3'],
        num_oov_buckets=0)
one_hot = feature_column.indicator_column(categorical)
feature_columns.append(one_hot)

categorical = tf.feature_column.categorical_column_with_vocabulary_list(
        key='f2', vocabulary_list=['x1','x2','x3','x4','x5','x6'],
        num_oov_buckets=5)
embedding = feature_column.embedding_column(categorical, dimension=10)
feature_columns.append(embedding)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(20, activation='relu'),
  layers.Dense(1,activation='softsign')
])
    
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

train = df_to_dataset(train)
val = df_to_dataset(val)

model.fit(train,
          validation_data=val,
          epochs=2)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model,Model
from tensorflow import feature_column
from tensorflow.keras import layers


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

###--------- sample dataset ---------
n = 10000
data = pd.DataFrame({'f1':np.random.choice(['x1','x2','x3'], n, replace=True),
                     'f2':np.random.choice(['x1','x2','x3','x4','x5','x6'], n, replace=True),
                     'target':np.random.choice([0,1], n, replace=True)})

train,val = train_test_split(data,test_size=0.2)
train,new_train = train_test_split(train,test_size=0.2)


###--------- define features columns ---------
feature_columns = []

categorical = tf.feature_column.categorical_column_with_vocabulary_list(
        key='f1', vocabulary_list=['x1','x2','x3'],
        num_oov_buckets=0)
one_hot = feature_column.indicator_column(categorical)
feature_columns.append(one_hot)

categorical = tf.feature_column.categorical_column_with_vocabulary_list(
        key='f2', vocabulary_list=['x1','x2','x3','x4','x5','x6'],
        num_oov_buckets=5)
embedding = feature_column.embedding_column(categorical, dimension=10)
feature_columns.append(embedding)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

###--------- define and compile model ---------
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(20, activation='relu'),
  layers.Dense(1,activation='softsign')
])
    
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


train = df_to_dataset(train)
val = df_to_dataset(val)
new_train = df_to_dataset(new_train)

#----------- train model first time - all its ok --------
model.fit(train,
          validation_data=val,
          epochs=2)

model.save('basic_model',save_format='tf')

###------------- first update, no errors -------------------------
model_update1 = tf.keras.models.load_model('basic_model')
model_update1.fit(new_train,
          validation_data=val,
          epochs=2)

model_update1.save('update1',save_format='tf')

###------------- 2nd update is problem and generate errors -------------------------
model_update2 = tf.keras.models.load_model('update1')
model_update2.fit(new_train,
          validation_data=val,
          epochs=2)