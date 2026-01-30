import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

from tensorflow.keras import initializers
from tensorflow.keras import layers


class CustomBatchNormalization(layers.Layer):
    def __init__(self, momentum=0.99, epsilon=1e-3,beta_initializer='zeros',
                 gamma_initializer='ones', moving_mean_initializer='zeros',
                 moving_range_initializer='ones',**kwargs):
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_range_initializer = (
            initializers.get(moving_range_initializer))
        
        super().__init__(**kwargs)
    
    def build(self,input_shape):
        dim = input_shape[-1]
        shape = (dim,)
        self.gamma = self.add_weight(shape=shape,
                             name='gamma',
                             initializer=self.gamma_initializer,trainable=True)
        self.beta = self.add_weight(shape=shape,
                            name='beta',
                            initializer=self.beta_initializer,
                                   trainable=True)
        
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        
        self.moving_range = self.add_weight(
            shape=shape,
            name='moving_range',
            initializer=self.moving_range_initializer,
            trainable=False)



    def call(self, inputs, training=None):
        input_shape = inputs.shape
        
        if not training:
            scaled = (inputs-self.moving_mean)/(self.moving_range+self.epsilon)
            return self.gamma*scaled + self.beta
        
        mean = tf.math.reduce_mean(inputs,axis=0)
        maxr = tf.math.reduce_max(inputs,axis=0)
        minr = tf.math.reduce_min(inputs,axis=0)
        
        range_diff = tf.math.subtract(maxr,minr)
        self.moving_mean = tf.math.add(self.momentum*self.moving_mean, (1-self.momentum)*mean)
        self.moving_range = tf.math.add(self.momentum*self.moving_range,(1-self.momentum)*range_diff)
        scaled = tf.math.divide(tf.math.subtract(inputs,mean),(range_diff+self.epsilon))
        return tf.math.add(tf.math.multiply(self.gamma,scaled),self.beta)
    
    def get_config(self):
        config = {
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_range_initializer':
                initializers.serialize(self.moving_range_initializer)
        }
        base_config = super(CustomBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape




##Below is the network 

inp = Input(shape=(64,))
batch_norm_1 = CustomBatchNormalization()(inp)
densout = Dense(128, activation='linear')(batch_norm_1)
densout = LeakyReLU(alpha=0.3)(densout)
for i in range (6):
    batch_norm_i = CustomBatchNormalization()(densout)
    densout = Dense(128, activation='linear')(batch_norm_i)
    densout = LeakyReLU(alpha=0.3)(densout)
batch_norm_out = CustomBatchNormalization()(densout)
out = Dense(64, activation='linear')(batch_norm_out)
Inp_RH_CBN = tf.keras.models.Model(inp, out)

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, LeakyReLU


class CustomBatchNormalization(layers.Layer):
    def __init__(self, momentum=0.99, epsilon=1e-3,beta_initializer='zeros',
                 gamma_initializer='ones', moving_mean_initializer='zeros',
                 moving_range_initializer='ones',**kwargs):
        super(CustomBatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_range_initializer = (
            initializers.get(moving_range_initializer))

    
    def build(self,input_shape):
        dim = input_shape[-1]
        shape = (dim,)
        self.gamma = self.add_weight(shape=shape,
                             name='gamma',
                             initializer=self.gamma_initializer,trainable=True)
        self.beta = self.add_weight(shape=shape,
                            name='beta',
                            initializer=self.beta_initializer,
                                   trainable=True)
        
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        
        self.moving_range = self.add_weight(
            shape=shape,
            name='moving_range',
            initializer=self.moving_range_initializer,
            trainable=False)



    def call(self, inputs,training=None):
        input_shape = inputs.shape
        
        if training == False:
            scaled = (inputs-self.moving_mean)/(self.moving_range+self.epsilon)
            return self.gamma*scaled + self.beta
        
        mean = tf.math.reduce_mean(inputs,axis=0)
        maxr = tf.math.reduce_max(inputs,axis=0)
        minr = tf.math.reduce_min(inputs,axis=0)
        
        range_diff = tf.math.subtract(maxr,minr)
        self.moving_mean = tf.math.add(self.momentum*self.moving_mean, (1-self.momentum)*mean)
        self.moving_range = tf.math.add(self.momentum*self.moving_range,(1-self.momentum)*range_diff)
        scaled = tf.math.divide(tf.math.subtract(inputs,mean),(range_diff+self.epsilon))
        return tf.math.add(tf.math.multiply(self.gamma,scaled),self.beta)
    
    def get_config(self):
        config = {
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_range_initializer':
                initializers.serialize(self.moving_range_initializer)
        }
        base_config = super(CustomBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape


inp = Input(shape=(64,))
batch_norm_1 = CustomBatchNormalization()(inp)
densout = Dense(128, activation='linear')(batch_norm_1)
densout = LeakyReLU(alpha=0.3)(densout)
for i in range (6):
    batch_norm_i = CustomBatchNormalization()(densout)
    densout = Dense(128, activation='linear')(batch_norm_i)
    densout = LeakyReLU(alpha=0.3)(densout)
batch_norm_out = CustomBatchNormalization()(densout)
out = Dense(64, activation='linear')(batch_norm_out)
Inp_RH_CBN = tf.keras.models.Model(inp, out)

def build(self,input_shape):
        dim = input_shape[-1]
        shape = (dim,)
        self.gamma = self.add_weight(shape=shape,
                             name='gamma',
                             initializer=self.gamma_initializer,trainable=True)
        self.beta = self.add_weight(shape=shape,
                            name='beta',
                            initializer=self.beta_initializer,
                                   trainable=True)
        
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        
        self.moving_range = self.add_weight(
            shape=shape,
            name='moving_range',
            initializer=self.moving_range_initializer,
            trainable=False)