import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

def NegLogProb(a, b, targ):
    return -tf.reduce_mean(tf.reduce_sum(tf.math.log(a*b) + (a-1)*tf.math.log(targ+1e-8) + (b-1)*tf.math.log((1-targ**a)+1e-8), axis=-1))

def NegLogProb(a, b, targ):
    return -tf.reduce_mean(tf.reduce_sum(tf.math.log(a*b) + (a-1)*tf.math.log(targ+1e-8) + (b-1)*tf.math.log((1-targ**a)+1e-8), axis=-1))
    
LossFunc = NegLogProb
def train_step(samples, targ):
    with tf.GradientTape() as tape:
        a,b = model(samples, training=True)
        loss = LossFunc(a, b, targ)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss, (a,b)

simintime = []
testintime = []
def train(epochs, batch_size=100, svwts=False):
    graph = train_step#tf.function(train_step)
    
    tot = intf.shape[0]
    steps = (tot//batch_size) if tot%batch_size==0 else (tot//batch_size)+1
    
    for m in range(epochs):
        tots = 0
        start_epoch = time()
        for n in range(steps):
            start_step = time()
            begin = n*batch_size
            end = (n+1)*(batch_size)
            
            targ = create_target(Pos[begin:end])
            sim, var = graph(intf[begin:end], targ)
            
            tots+=sim.numpy()*(end-begin)
            sys.stdout.write("\rStep %d/%d; -log_prob: %8.1f; Time: %.1f s"%(n+1, steps, sim, time()-start_step))
        final_sim = tots/tot
        simintime.append(final_sim)

def train(epochs, batch_size=100, svwts=False):
    graph = tf.function(train_step)
    
    tot = intf.shape[0]
    steps = (tot//batch_size) if tot%batch_size==0 else (tot//batch_size)+1
    
    for m in range(epochs):
        tots = 0
        start_epoch = time()
        for n in range(steps):
            start_step = time()
            begin = n*batch_size
            end = (n+1)*(batch_size)
            
            targ = create_target(Pos[begin:end])
            sim, var = graph(intf[begin:end], targ)
            
            tots+=sim.numpy()*(end-begin)
            sys.stdout.write("\rStep %d/%d; -log_prob: %8.1f; Time: %.1f s"%(n+1, steps, sim, time()-start_step))
        final_sim = tots/tot
        simintime.append(final_sim)

import numpy as np
import os
# import re
import sys
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.activations as A
from time import time
import matplotlib.pyplot as plt
plt.close('all')
from matplotlib.ticker import MultipleLocator

targ = tf.constant(
    [[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.20933868,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.2082187 , 0.        ,
        0.18404533, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.37072948, 0.        ,
        0.        , 0.        , 0.        , 0.31289002, 0.        ,
        0.        , 0.        , 0.915976  , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.23793682, 0.        , 0.        ,
        0.        , 0.4674082 , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.18847302,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.2036586 , 0.17317393, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.29346868,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.40215385, 0.        , 0.16911463, 0.        ,
        0.        , 0.        , 0.24255644, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.24563654, 0.        , 0.        , 0.        , 0.33256173,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.30373514, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.2429015 , 0.15512827, 0.        , 0.        ,
        0.        , 0.41071343, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.29341364, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.21696085, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.1478434 , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.35621345, 0.        , 0.        , 0.        ,
        0.        , 0.15217698, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.38775736, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.19092496,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.18351643, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.5620028 ,
        0.3221243 , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.18488021, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.14928055, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.28359178,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 1.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.17243865, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.4688842 , 0.30614266, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.18065585, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.1995775 , 0.        , 0.        , 0.        , 0.        ,
        0.59800917, 0.        , 0.        , 0.        , 0.        ,
        0.2842905 , 0.        , 0.        , 0.98819506, 0.20961763,
        0.        , 0.        , 0.        , 0.        , 0.30928153,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ]], dtype=tf.float32
)

sample = tf.constant([[19,  4, 15, 15,  4, 16, 16,  6,  8,  8, 20, 21, 21, 21, 21, 21,
        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
        21, 21, 21, 21, 21, 21, 21, 21, 21]], dtype=tf.int32)

class ResBlock2(L.Layer):
    def __init__(self, filters, KS=3, conv_type=L.Conv1D, act=A.relu):
        super(ResBlock2, self).__init__()
        self.filters = filters
        self.conv_type = conv_type
        self.norm1 = L.BatchNormalization()
        self.norm2 = L.BatchNormalization()
        self.conv1 = conv_type(filters, KS, 1, padding='SAME', use_bias=False) # bias removed after running
        self.conv2 = conv_type(filters, KS, 1, padding='SAME', use_bias=False)
        self.act = act
    def build(self, x):
        self.shortcut = self.conv_type(self.filters, 1, 1) if self.filters!= x[-1] else A.linear
    def call(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(L.Add()([self.shortcut(x), self.norm2(self.conv2(out))]))
        return out

def CustomBlock(feat_map, rang, feat_out=128, residual=True, pad='SAME', alpha=0.0):
    rang = tf.range(*rang)
    out = tf.concat([L.Conv1D(feat_out, int(m), 1, padding=pad)(feat_map) for m in rang], axis=-1)
    out = L.BatchNormalization()(out)
    if residual:
        shortcut = L.Conv1D(out.shape[-1], 1, 1) if feat_map.shape[-1]!=out.shape[-1] else A.linear 
        out = L.Add()([shortcut(feat_map), out])
    return A.relu(out, alpha)

upsample = lambda x: tf.reshape(tf.tile(tf.expand_dims(x, axis=2), [1,1,2,1]), (-1, 2*x.shape[1], x.shape[-1]))

def Model_beta(seq_len, 
               AA_types, 
               out_dim,
               floors=(1,1),
               filtfirst=64,
               rang=(2,10,1),
               outin=(3,5), 
               filtmid=(150, 200), 
               filtlast=200):
    inp = L.Input((seq_len,), dtype=tf.int32)
    
    out = tf.one_hot(inp, AA_types) # Original
    out = CustomBlock(out, (3,4,1), feat_out=64, residual=False, alpha=0.0)
    
    outer,inner = outin
    filts = np.linspace(filtmid[0], filtmid[1], outer, dtype='int')
    for m in range(outer):
        out = upsample(out) if m>0 else out
        for n in range(inner):
            filters = filts[m]
            ks = 3
            out = ResBlock2(filters, ks)(out)
    
    out = CustomBlock(out, (1,2,1), filtlast, residual=False)
    
    a = L.Conv1D(out_dim, 1, 1, activation='relu')(out)
    a = L.GlobalAveragePooling1D()(a)
    a = tf.squeeze(a)+floors[0]
    b = L.Conv1D(out_dim, 1, 1, activation='linear')(out)
    b = L.GlobalAveragePooling1D()(b)
    b = tf.squeeze(a)+floors[1]
    
    return tf.keras.Model(inputs=inp, outputs=[a,b])

model = Model_beta(41, 22, 910, 2*(1+1e-6,))
opt = tf.keras.optimizers.Adam(3e-4)

def NegLogProb(a, b, targ):
    return -tf.reduce_mean(tf.reduce_sum(tf.math.log(a*b) + (a-1)*tf.math.log(targ+1e-8) + (b-1)*tf.math.log((1-targ**a)+1e-8), axis=-1))

def train_step(samples, targ):
    with tf.GradientTape() as tape:
        a,b = model(samples, training=True)
        loss = NegLogProb(a, b, targ)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss, (a,b)

outeager,_ = train_step(sample, targ)

print(outeager)

graph = tf.function(train_step)

outgraph,_ = graph(sample, targ)

print(outgraph)

import tensorflow as tf

tf.config.optimizer.set_experimental_options(
    {'constant_folding': True})  # Returns correct results with False

@tf.function
def NegLogProb(a, b, targ):
    return tf.add((1 - tf.pow(targ, a)), 1e-7)

NegLogProb(tf.constant([1.0]), tf.constant([1.0]), tf.constant(1.0))