import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
from time import time

tf.random.set_seed(0)
lb = tf.constant([0,0], dtype=np.float32)
ub = tf.constant([1,1], dtype=np.float32)
N_0     = 4092
Nepochs = 1500
lr      = 0.001

def fun_u_0(xx):
    c_0  = 0.5*(lb+ub)
    rr   = 0.25*tf.reduce_min(ub-lb)    
    dsq  = tf.math.reduce_sum( (xx-c_0)*(xx-c_0),axis=1)
    return(tf.where( dsq<=rr*rr, 1.0, 0.0) )

def init_model(num_hidden_layers=2, num_neurons_per_layer=64):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(2,)) )  
    #scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
    #model.add(scaling_layer)    
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer, activation=tf.keras.layers.LeakyReLU(
        ),kernel_initializer="glorot_uniform")  )    
    # Output is one-dimensional
    model.add(tf.keras.layers.Dense(1,kernel_initializer="glorot_uniform"))    
    return model

X_data = tf.random.uniform((N_0,2), lb, ub, dtype=np.float32)
u_data = fun_u_0(X_data)

train_len = int(0.8*X_data.shape[0])
X_train = X_data[:train_len]
u_train = u_data[:train_len]
X_test  = X_data[train_len:]
u_test  = u_data[train_len:]

# This version works:
def my_loss(u_true, u_pred):
    return tf.math.reduce_mean(tf.math.square(u_true - u_pred))

model_0 = init_model(num_hidden_layers=2, num_neurons_per_layer=64)
optim_0 = tf.keras.optimizers.Adam(learning_rate=lr)
model_0.compile(loss=my_loss, optimizer=optim_0)
model_0.summary()
history_0 = model_0.fit(X_train,u_train,validation_data=(X_test.numpy(),u_test.numpy()), verbose=0,epochs=Nepochs,batch_size=X_train.shape[0])

# This version does not work:
def compute_loss(model, X_data, u_data):
    u_pred = model(X_data)
    loss = tf.math.reduce_mean(tf.math.square(u_data - u_pred))
    return loss

@tf.function
def training(model,optim,X_train,u_train,X_test=None,u_test=None):
    if X_test is not None:
        validation_loss  = compute_loss(model, X_test, u_test )
    else:
        validation_loss = None        
    with tf.GradientTape(persistent=True) as tape:
        theta = model.trainable_variables
        tape.watch(theta)
        loss = compute_loss(model, X_train, u_train )
    grad_theta = tape.gradient(loss, theta)
    optim.apply_gradients(zip(grad_theta, model.trainable_variables))
    return loss,validation_loss

model_G = init_model(num_hidden_layers=2, num_neurons_per_layer=64)
optim_G = tf.keras.optimizers.Adam(learning_rate=lr)
model_G.summary()

hist = {'val_loss':[],'loss':[]}
for i in range(Nepochs+1):
        loss, val_loss = training(model_G,optim_G,X_train,u_train,X_test,u_test)
        # Append current loss to hist
        hist['loss'].append(loss.numpy())
        hist['val_loss'].append(val_loss.numpy())
        # Output current loss after 50 iterates
        if val_loss is not None:
          print('It {:05d}: loss = {:10.8e}, validation loss = {:10.8e}'.format(i,loss,val_loss))
        else:
          print('It {:05d}: loss = {:10.8e}'.format(i,loss))

u_train = tf.expand_dims(u_train,axis=-1)
u_test = tf.expand_dims(u_test,axis=-1)