from tensorflow import keras
from tensorflow.keras import models

from os import walk
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import random
from scipy import signal
import os
import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GRU, LSTM, TimeDistributed, RepeatVector, Bidirectional
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
        self.times = []
        self.totaltime = time.time()
        
    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime
    
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
        
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        self.times.append(time.time() - self.epoch_time_start)
   
    def loss_plot(self, loss_type):
        acc =[]
        loss = []
        val = []
        iters = range(len(self.losses[loss_type]))
        # acc
        acc.extend(self.accuracy[loss_type])
        # loss
        print("loss = ",self.losses[loss_type])
        loss.extend(self.losses[loss_type])
        # val
        print("val = ",self.losses[loss_type])
        val.extend(self.val_acc[loss_type])
        return(acc, loss,val)
time_callback = TimeHistory()
      
def buildManyToOneModel(shape):

    model = tf.keras.models.Sequential([
        GRU(32, input_dim = shape[2], input_length = shape[1], return_sequences = True),
        GRU(64, return_sequences = True),
        GRU(128, return_sequences = False),
        #LSTM(16, return_sequences = True),
        #LSTM(16, return_sequences = False),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='mse', optimizer = 'adam', metrics=['acc'])
    model.summary()
    return model

def slice_(data, node_num):
    total = float(data.shape[0])
    store = []
    if node_num == 1:
        store.append(data[0:int(total),:])
        store.append([0])
        store.append([0])
    elif node_num == 2:
        slice_index = int(total / 2)
        store.append(data[0:slice_index, :])
        store.append(data[slice_index:int(total), :])
        store.append([0])
    elif node_num == 3:
        slice_index = int(total / 3)
        store.append(data[0:slice_index, :])
        store.append(data[slice_index:2*slice_index, :])
        store.append(data[2*slice_index:int(total), :])
    return store

def train():
    print("TensorFlow version: ", tf.__version__)
    tf_config = os.environ.get('TF_CONFIG', '{}')
    print("TF_CONFIG %s", tf_config)
    tf_config_json = json.loads(tf_config)
    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    print("cluster={} job_name={} task_index={}}", cluster, job_name, task_index)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=tf.distribute.experimental.CollectiveCommunication.RING)
    print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    data0 = np.load('/app/data/2017_360w_data_0.npy')
    data1 = np.load('/app/data/2017_360w_data_1.npy')
    data0 = data0[:,:,0]
    data1 = data1[:,:,0]
    print("data0:",data0.shape)
    print("data1:",data1.shape)
    a1 = np.array(data0)[0 : int(data0.shape[0]*0.7), :]
    a2 = np.array(data1)[0 : int(data1.shape[0]*0.7), :]
    a3 = np.array(data0)[int(data0.shape[0]*0.7) : data0.shape[0], :]
    a4 = np.array(data1)[int(data1.shape[0]*0.7) : data1.shape[0], :]
    X_train = np.concatenate((a1, a2), axis=0) 
    X_val = np.concatenate((a3, a4), axis=0) 

    b1 = np.zeros((a1.shape[0], 1))
    b2 = np.ones((a2.shape[0], 1))

    b3 = np.zeros((a3.shape[0], 1))

    b4 = np.ones((a4.shape[0], 1))

    Y_train = np.concatenate((b1, b2), axis=0)
    Y_val = np.concatenate((b3, b4), axis=0)
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    Y_val = Y_val.astype(np.float32)
    X_train = X_train[:,:,np.newaxis]

    X_val = X_val[:,:,np.newaxis]
    print(X_train.shape)
    print(Y_train.shape)
    print(X_val.shape)
    print(Y_val.shape)
    print(type(X_train)) 
    print(type(Y_train))    
    BUFFER_SIZE = X_train.shape[0]
 
    BATCH_SIZE_PER_REPLICA = 5000
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA  * (strategy.num_replicas_in_sync-1)
    
    #train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    #test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset) 
   
    if BUFFER_SIZE % GLOBAL_BATCH_SIZE != 0:
        parallel_steps =  X_train.shape[0] // GLOBAL_BATCH_SIZE + 1
        a =  X_val.shape[0] // GLOBAL_BATCH_SIZE + 1
    else:
        parallel_steps =  X_train.shape[0] // GLOBAL_BATCH_SIZE
        a =  X_val.shape[0] // GLOBAL_BATCH_SIZE         
    print(parallel_steps) 
    t2 = time.time()
    with strategy.scope():
         train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).repeat().shuffle(buffer_size=5000000).batch(GLOBAL_BATCH_SIZE)
         #test_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(GLOBAL_BATCH_SIZE) 
         options = tf.data.Options()
         options.experimental_distribute.auto_shard_policy = \
                                        tf.data.experimental.AutoShardPolicy.DATA
         train_dataset = train_dataset.with_options(options)    
         #test_dataset = test_dataset.with_options(options)        
         multi_worker_model = buildManyToOneModel(X_train.shape)    

    #history = multi_worker_model.fit(train_dataset, epochs=30, validation_data=test_dataset,steps_per_epoch=parallel_steps,validation_steps=a,callbacks=[time_callback])
    history = multi_worker_model.fit(train_dataset, epochs=30,steps_per_epoch=parallel_steps,callbacks=[time_callback])
    t3 = time.time()    
    print ("It cost ", t3 - t2, " seconds")


    accuracy, loss,val = time_callback.loss_plot('epoch')
    print("totaltime:%.4f"%time_callback.totaltime)
    for i in range(len(accuracy)):
        print("acc: %.4f, loss: %.4f,val:%.4f -----epoch:%d" %(accuracy[i], loss[i],val[i],i+1))
    totaltime='%.2f'%time_callback.totaltime
    ttime= t3 - t2
    traningtime='%.2f'%ttime
    maxval='%.2f'%max(val)
    save_dir = '/app/data'
    f = open(save_dir+"/720_1.txt", "a")
    f.write("\ntotaltime:{},traningtime:{},val:{}".format(totaltime,traningtime,maxval))
    f.close()
if __name__ == '__main__':
    train()