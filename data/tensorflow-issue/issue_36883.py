import numpy as np
import math
import random
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras.layers import LeakyReLU
import tensorflow as tf
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import mse

### create a custom layer

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


### create your model
inp = Input(shape=(4,))
batch_norm_1 = CustomBatchNormalization(dynamic=True)(inp)
densout = Dense(24, activation='linear')(batch_norm_1)
densout = LeakyReLU(alpha=0.3)(densout)
batch_norm_2 = CustomBatchNormalization(dynamic=True)(densout)
densout = Dense(128, activation='linear')(batch_norm_2)
densout = LeakyReLU(alpha=0.3)(densout)
batch_norm_out = CustomBatchNormalization(dynamic=True)(densout)
out = Dense(5, activation='linear')(batch_norm_out)
test_nw = tf.keras.models.Model(inp, out)

##compile it
test_nw.compile(tf.keras.optimizers.Adam(), loss=mse,experimental_run_tf_function=False)

path_HDF5 = 'PATH_TO_SAVE_THIS_MODEL'
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save_RH = ModelCheckpoint(path_HDF5,save_best_only=True, monitor='val_loss', mode='min')

X = np.random.randn(4,4)
y = np.random.randn(4,5)
X_val = np.random.randn(4,4)
y_val = np.random.randn(4,5)
test_nw.fit(X,y,batch_size=4, epochs=10,validation_data = (X_val,y_val),callbacks=[earlyStopping, mcp_save_RH] )

######## Now restart the kernel and load the model
dict_lay = {'CustomBatchNormalization':CustomBatchNormalization}
mod = load_model(path_HDF5,custom_objects=dict_lay)