import math
import random

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow import keras as k
import numpy as np

##making empty directories
import os
os.makedirs('r_data',exist_ok=True)
os.makedirs('r_savedir',exist_ok=True)

#Preparing the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_ = pd.DataFrame(x_train.reshape(60000,-1),columns = ['col_'+str(i) for i in range(28*28)])
x_test_ = pd.DataFrame(x_test.reshape(10000,-1),columns = ['col_'+str(i) for i in range(28*28)])
x_train_['col_cat1'] = [np.random.choice(['a','b','c','d','e','f','g','h','i']) for i in range(x_train_.shape[0])]
x_test_['col_cat1'] = [np.random.choice(['a','b','c','d','e','f','g','h','i','j']) for i in range(x_test_.shape[0])]
x_train_['col_cat2'] = [np.random.choice(['a','b','c','d','e','f','g','h','i']) for i in range(x_train_.shape[0])]
x_test_['col_cat2'] = [np.random.choice(['a','b','c','d','e','f','g','h','i','j']) for i in range(x_test_.shape[0])]
x_train_[np.random.choice([True,False],size = x_train_.shape,p=[0.05,0.95]).reshape(x_train_.shape)] = np.nan
x_test_[np.random.choice([True,False],size = x_test_.shape,p=[0.05,0.95]).reshape(x_test_.shape)] = np.nan
x_train_.to_csv('r_data/x_train.csv',index=False)
x_test_.to_csv('r_data/x_test.csv',index=False)
pd.DataFrame(y_train).to_csv('r_data/y_train.csv',index=False)
pd.DataFrame(y_test).to_csv('r_data/y_test.csv',index=False)

#**THE MAIN LAYER THAT WE ARE TALKING ABOUT**
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import feature_column
import os

class NUM_TO_DENSE(layers.Layer):
    def __init__(self,num_cols):
        super().__init__()
        self.keys = num_cols
        self.keys_all = self.keys+[str(i)+'__nullcol' for i in self.keys]
#     def get_config(self):

#         config = super().get_config().copy()
#         config.update({
#             'keys': self.keys,
#             'keys_all': self.keys_all,
#         })
#         return config
    def build(self,input_shape):
        def create_moving_mean_vars():
            return tf.Variable(initial_value=0.,shape=(),dtype=tf.float32,trainable=False)
        self.moving_means_total = {t:create_moving_mean_vars() for t in self.keys}
        self.layer_global_counter = tf.Variable(initial_value=0.,shape=(),dtype=tf.float32,trainable=False)

    def call(self,inputs, training = True):
        null_cols = {k:tf.math.is_finite(inputs[k]) for k in self.keys}
        current_means = {}
        def compute_update_current_means(t):
            current_mean = tf.math.divide_no_nan(tf.reduce_sum(tf.where(null_cols[t],inputs[t],0.),axis=0),\
                                  tf.reduce_sum(tf.cast(tf.math.is_finite(inputs[t]),tf.float32),axis=0))
            self.moving_means_total[t].assign_add(current_mean)
            return current_mean
        
        if training:
            current_means = {t:compute_update_current_means(t) for t in self.keys}
            outputs = {t:tf.where(null_cols[t],inputs[t],current_means[t]) for t in self.keys}
            outputs.update({str(k)+'__nullcol':tf.cast(null_cols[k],tf.float32) for k in self.keys})
            self.layer_global_counter.assign_add(1.)
        else:
            outputs = {t:tf.where(null_cols[t],inputs[t],(self.moving_means_total[t]/self.layer_global_counter))\
                       for t in self.keys}
            outputs.update({str(k)+'__nullcol':tf.cast(null_cols[k],tf.float32) for k in self.keys})
        return outputs


class PREPROCESS_MONSOON(layers.Layer):
    def __init__(self,cat_cols_with_unique_values,num_cols):
        '''cat_cols_with_unqiue_values: (dict) {'col_cat':[unique_values_list]}
        num_cols: (list) [num_cols_name_list]'''
        super().__init__()
        self.cat_cols = cat_cols_with_unique_values
        self.num_cols = num_cols
#     def get_config(self):

#         config = super().get_config().copy()
#         config.update({
#             'cat_cols': self.cat_cols,
#             'num_cols': self.num_cols,
#         })
#         return config
    def build(self,input_shape):
        self.ntd = NUM_TO_DENSE(self.num_cols)
        self.num_colnames = self.ntd.keys_all
        self.ctd = {k:layers.DenseFeatures\
                    (feature_column.embedding_column\
                     (feature_column.categorical_column_with_vocabulary_list\
                      (k,v),tf.cast(tf.math.ceil(tf.math.log(tf.cast(len(self.cat_cols[k]),tf.float32))),tf.int32).numpy()))\
                   for k,v in self.cat_cols.items()}
        self.cat_colnames = [i for i in self.cat_cols]
        self.dense_colnames = self.num_colnames+self.cat_colnames
    def call(self,inputs,training=True):
        dense_num_d = self.ntd(inputs,training=training)
        dense_cat_d = {k:self.ctd[k](inputs) for k in self.cat_colnames}
        
        dense_num = tf.stack([dense_num_d[k] for k in self.num_colnames],axis=1)
        dense_cat = tf.concat([dense_cat_d[k] for k in self.cat_colnames],axis=1)
        dense_all = tf.concat([dense_num,dense_cat],axis=1)
        return dense_all

##Inputs
label_path = 'r_data/y_train.csv'
data_path = 'r_data/x_train.csv'
max_epochs = 100
batch_size = 32
shuffle_seed = 42

##Creating layer inputs
dfs = pd.read_csv(data_path,nrows=1)
cdtypes_x = dfs.dtypes
nc = list(dfs.select_dtypes(include=[int,float]).columns)
oc = list(dfs.select_dtypes(exclude=[int,float]).columns)
cdtypes_y = pd.read_csv(label_path,nrows=1).dtypes
dfc = pd.read_csv(data_path,usecols=oc)
ccwuv = {i:list(pd.Series(dfc[i].unique()).dropna()) for i in dfc.columns}
preds_name = pd.read_csv(label_path,nrows=1).columns

##creating datasets
dataset = tf.data.experimental.make_csv_dataset(
    'r_data/x_train.csv',batch_size, column_names=cdtypes_x.index,prefetch_buffer_size=1,
shuffle=True,shuffle_buffer_size=10000,shuffle_seed=shuffle_seed)
labels = tf.data.experimental.make_csv_dataset(
    'r_data/y_train.csv',batch_size, column_names=cdtypes_y.index,prefetch_buffer_size=1,
shuffle=True,shuffle_buffer_size=10000,shuffle_seed=shuffle_seed)
dataset = tf.data.Dataset.zip((dataset,labels))

##CREATING NETWORK
p = PREPROCESS_MONSOON(cat_cols_with_unique_values=ccwuv,num_cols=nc)

indict = {}
for i in nc:
    indict[i] = k.Input(shape = (), name=i,dtype=tf.float32)
for i in ccwuv:
    indict[i] = k.Input(shape=(), name=i,dtype=tf.string)
x = p(indict)
x = l.BatchNormalization()(x)
x = l.Dense(10,activation='relu',name='dense_1')(x)
predictions = l.Dense(10,activation=None,name=preds_name[0])(x)
model = k.Model(inputs=indict,outputs=predictions)

##Compiling model
model.compile(optimizer=k.optimizers.Adam(),
              loss=k.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

##callbacks
log_dir = './tensorboard_dir/no_config'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

## Fit model on training data
history = model.fit(dataset,
                    batch_size=64,
                    epochs=30,
                    steps_per_epoch=5,
                    validation_split=0.,
                   callbacks = [tensorboard_callback])

#saving the model
tf.saved_model.save(model,'r_savedir')
#loading the model
model = tf.saved_model.load('r_savedir')

##Predicting on loaded model
for i in dataset:
    print(model(i[0],training=False))
    break