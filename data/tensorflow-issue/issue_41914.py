import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

class Rand(Activation):                
     def __init__(self, activation, **kwargs):
        super(Rand, self).__init__(activation, **kwargs)
        self.__name__ = 'rand'

 
def rand(x):
    result = tf.Variable(tf.cond(tf.random.uniform(shape=[1])[0] > tf.Variable(x), 1, 0))
    return result

from __future__ import print_function
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils 
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras import regularizers
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_score , recall_score
from sklearn.metrics import confusion_matrix
from keras.callbacks import Callback, LearningRateScheduler
import math
from keras.models import load_model
import csv
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.optimizers import adam
from keras.callbacks import Callback, LearningRateScheduler
from keras import backend as K

#activation function
class Rand(Activation):             # Take care of Rand and rand in the following few lines
    
    def __init__(self, activation, **kwargs):
        super(Rand, self).__init__(activation, **kwargs)
        self.__name__ = 'rand'

  
def rand(x):
    # r = np.random.uniform(low=0.0, high=1.0, size=None)
    # bool = tf.Variable(x<r)
    # result = init_bias = tf.Variable(init_bias,validate_shape=False)
    result = tf.Variable(tf.cond(tf.random.uniform(shape=[1])[0] > tf.Variable(x), 1, 0))
    print ("asdasdasdasD", result)
    return result
    # if x is not None:
        # print ("yes")
    # if x<r:
        # return int(0)
    # else:
        # return int(1)

get_custom_objects().update({'rand': Rand(rand)})

path_data_original = "/home/tntech.edu/miibrahem42/GAN_Paper/Defense/"

# load the data
def data():
    no_samples = 200000 #number of zero of one samples
    last_col_indx = 100
    data = pd.read_csv('Dataset.csv', sep=',', index_col=False, header=None)
    
    # # take n balanced samples
    # s0 = data[last_col_indx][data[last_col_indx].eq(0)].sample(no_samples).index
    # s1 = data[last_col_indx][data[last_col_indx].eq(1)].sample(no_samples).index 
    # data = data.loc[s0.union(s1)]
    # # data = data.reindex(np.random.permutation(data.index))
    # data = data.sample(frac=1)
    # # print (len(list(data.to_numpy())[0]))
    # # print ((list(data.to_numpy())))
    
    X_res = data.iloc[:,:data.shape[1]-1].to_numpy() #400,000
    print ("total data shape: ",X_res.shape)
    Y_res = data.iloc[:,data.shape[1]-1]
    print ("total label shape: ",Y_res.shape)
    
    ada = RandomUnderSampler(ratio='majority',random_state=42)
    x_train, y_train = ada.fit_sample(X_res,Y_res) # over-sampled data
    print ("sum of resulted labels is: ", sum(y_train))

    xtr, x_valid, ytr, y_valid = train_test_split(x_train, y_train, test_size=0.3)

    xtr = xtr.reshape(-1,xtr.shape[1],1)
    x_valid = x_valid.reshape(-1,x_valid.shape[1],1)
    
    nb_classes = 2

    print ("Number of training samples is: ",xtr.shape[0])
    print ("Number of test samples is: ",x_valid.shape[0])

    # ytr = to_categorical(ytr, nb_classes)
    # y_valid = to_categorical(y_valid, nb_classes)

    print ("label has a shape of: ", y_valid.shape,ytr.shape)
    return xtr,ytr,x_valid,y_valid

lr = 0.0001 

def scheduler(epoch):
  if epoch < 8:
    print (lr)
    return lr
  else:
    return lr * math.exp(0.1 * (4 - epoch))
learning_rte = LearningRateScheduler(scheduler)

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        # loss, acc = self.model.evaluate(x, y, verbose=1)
        # print('\nTesting loss: {}, acc: {}\n'.format(loss, acc*100))
 

xtr,ytr,x_valid,y_valid=data()

# input dimensions

# Train a RNN model without optimization

batch_size = 400
num_classes = 2
epochs = 1  #should be set

input_shape = (100,1)

print (xtr.shape)
print (x_valid.shape)

# the monitoring parameter should be the same on both earlystopping and modelcheckpoint in case of classification problems
stop_training = EarlyStopping( monitor='val_accuracy', mode='max', verbose=1, patience=15) #monitor='val_loss', mode='min'
best_model_save = ModelCheckpoint(path_data_original+'best_model_RNN_rand.h5',monitor='val_accuracy', mode='max', verbose=1
, save_best_only=True)

model = Sequential()
model.add(Conv1D(150, kernel_size=50, activation='relu', input_shape=(input_shape)))
model.add(MaxPooling1D(pool_size=4))
model.add(GRU(200))
model.add(Dropout(0.25))
# model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='relu', use_bias=False))
model.add(Dense(1, activation='rand', use_bias=False))  
print (model.summary())
# model.layers[8].trainable = False  

# # weights_of_last_layer = list([np.array([[1 , 1 ],
       # # [0, 0]], dtype='float32'), np.array([0., 0.], dtype='float32')])

# weights_of_last_layer = list([np.array([[1] ,[ 0 ]], dtype='float32'), np.array([0.], dtype='float32')])
       
# model.layers[8].set_weights(weights_of_last_layer)  
  
model.compile(loss='mean_squared_error', optimizer=adam(lr=lr), metrics=['accuracy'])  #categorical_crossentropy if binary classification
model.fit(xtr, ytr,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, 
         validation_split=0.1, callbacks=[TestCallback((x_valid, y_valid)), best_model_save, 
         stop_training, learning_rte])