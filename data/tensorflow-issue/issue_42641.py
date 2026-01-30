import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

X_train.shape=1200,18,15 
y_train.shape=1200,18,1

def twds_model(layer1=32, layer2=32, layer3=16, dropout_rate=0.5, optimizer='Adam'
                    , learning_rate=0.001, activation='relu', loss='mse'):#, n_jobs=1):layer3=80, 
    
    model = Sequential()
    model.add(Bidirectional(GRU(layer1, return_sequences=True),input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(AveragePooling1D(2))
    model.add(Conv1D(layer2, 3, activation=activation, padding='same', 
               name='extractor'))
    model.add(Flatten())
    model.add(Dense(layer3,activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=optimizer,loss=loss)
    return model

twds_model=twds_model()
print(twds_model.summary())

model_twds=KerasRegressor(build_fn=twds_model, batch_size=144,epochs=6)#12

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # Set to -1 if CPU should be used CPU = -1 , GPU = 0

gpus = tf.config.experimental.list_physical_devices('GPU')
cpus = tf.config.experimental.list_physical_devices('CPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
elif cpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        logical_cpus= tf.config.experimental.list_logical_devices('CPU')
        print(len(cpus), "Physical CPU,", len(logical_cpus), "Logical CPU")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#from __future__ import print_function, division


import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import sys
import os

import tsaug
from tsaug.visualization import plot
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Convolve, AddNoise, Dropout, Pool, Resize

import statsmodels
import datawig
import impyute

import missingpy
from missingpy import KNNImputer,MissForest

from impyute.imputation.cs import mice
from datawig import SimpleImputer
from statsmodels import robust
from operator import itemgetter,attrgetter
from functools import partial
from scipy import stats

from pylab import rcParams
from tpot import TPOTRegressor

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, LabelEncoder, RobustScaler, QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict, cross_validate, GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, make_scorer,median_absolute_error, mean_absolute_error,max_error,explained_variance_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from kerastuner import HyperModel



from tensorflow.python.keras.layers import InputLayer, TimeDistributed, Lambda, Dense, Dot, Reshape,Concatenate, Embedding, Activation, Conv1D, Conv2D, Cropping2D, MaxPooling2D, Flatten, Dropout, LSTM, GRU, Bidirectional, Input, LeakyReLU,Conv2DTranspose, ZeroPadding2D, ZeroPadding1D, UpSampling2D, UpSampling1D,multiply,AveragePooling1D # components of network
from tensorflow.python.keras.models import Model, Sequential # type of model
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.optimizers import Adam, RMSprop, SGD, Nadam, Adadelta, Adamax
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard



from ctgan import CTGANSynthesizer

#import tensorflow.keras.backend as K
from tensorflow.python.keras.backend import expand_dims, squeeze

from tqdm import tqdm


X_full = np.random.rand(1200,18,15)
y_full = np.random.rand(1200,18 )


y_test=np.array(pd.DataFrame(y_full).tail(round(0.10001671123*y_full.shape[0])))
y_train=np.array(pd.DataFrame(y_full).head(round((1-0.10001671123)*y_full.shape[0])))


X_test=np.array(pd.DataFrame(X_full.reshape(X_full.shape[0],X_full.shape[1]*X_full.shape[2])).tail(round(0.10001671123*X_full.shape[0]))).reshape(round(0.10001671123*X_full.shape[0]),X_full.shape[1],X_full.shape[2])
X_train=np.array(pd.DataFrame(X_full.reshape(X_full.shape[0],X_full.shape[1]*X_full.shape[2])).head(round(1-0.10001671123*X_full.shape[0]-1))).reshape(round((1-0.10001671123)*X_full.shape[0]),X_full.shape[1],X_full.shape[2])

Train_Len=len(X_train)
Test_Len=len(X_test)

print(y_train.shape,y_test.shape,X_train.shape,X_test.shape)


X_test_scaler = preprocessing.StandardScaler()
y_test_scaler = preprocessing.StandardScaler()
X_train_scaler = preprocessing.StandardScaler()
y_train_scaler = preprocessing.StandardScaler()

y_test=y_test_scaler.fit_transform(y_test)
y_train=y_train_scaler.fit_transform(y_train)

X_test=X_test_scaler.fit_transform(X_test.reshape(Test_Len*n_steps,Set_Width-1))
X_train=X_train_scaler.fit_transform(X_train.reshape(Train_Len*n_steps,Set_Width-1))

X_test=X_test.reshape(Test_Len,n_steps,Set_Width-1)
X_train=X_train.reshape(Train_Len,n_steps,Set_Width-1)



#########Model and SKlearn Cross_Val_Predict

def twds_model(layer1=32, layer2=32, layer3=16, dropout_rate=0.5, optimizer='Adam'
                    , learning_rate=0.001, activation='relu', loss='mse'):#, n_jobs=1):layer3=80, 
    
    model = Sequential()
    model.add(Bidirectional(GRU(layer1, return_sequences=True),input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(AveragePooling1D(2))
    model.add(Conv1D(layer2, 3, activation=activation, padding='same', 
               name='extractor'))
    model.add(Flatten())
    model.add(Dense(layer3,activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=optimizer,loss=loss)
    return model

twds_model=twds_model()
print(twds_model.summary())


def CustomVarious(y_true, y_pred):
    y_true=y_true.reshape(len(y_true[:,1])*Heisei_TR.shape[1],)
    
    if np.isnan(y_pred).any():
        result=-1000000
        MAD= 1000000
    else:
        y_pred=y_pred.reshape(len(y_pred[:,1])*Heisei_TR.shape[1],)
        MAD=median_absolute_error(y_true, y_pred)

        print(MAD)

        
        

    return MAD

scorer = make_scorer(CustomVarious, greater_is_better=False)

model_twds=KerasRegressor(build_fn=twds_Model, batch_size=256,epochs=6)


############# PLACE OF THE ERROR ############
twds_Pred=cross_val_predict(model_twds, 
               X_train, 
               y_train, 
               n_jobs=1, 
               cv=4, 
               verbose=2)

X_train = np.random.rand(1200,18,15)
X_train = np.random.rand(1200,18,1 )