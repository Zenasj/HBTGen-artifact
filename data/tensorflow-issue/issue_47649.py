import math
import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int = None,
        output_dim: int = 512,
        context_dim: int = 5,
        stride: int = 1,
        dilation: int = 1,
        kernel_initializer='glorot_uniform'
    ):
        super(MyLayer, self).__init__()
        self.input_dim = input_dim
        self.conv = tf.keras.layers.Conv1D(filters=output_dim,
                                           kernel_size=context_dim,
                                           strides=stride,
                                           dilation_rate=dilation,
                                           kernel_initializer=kernel_initializer)

    def call(self, x):
        batch_size, num_frames, num_feats = x.shape
        # if self.input_dim:
        #     assert self.input_dim == num_feats
        return self.conv(x)

class Reg(tf.keras.layers.Layer):
    def __init__(self,dropout_rate=0.2):
        super(Reg,self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.do = tf.keras.layers.Dropout(rate=dropout_rate)
    def call(self,x,training=True):
        x = self.bn(x,training=training)
        return self.do(x,training=training)
        
class MyModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim,dropout_rate=0.2,batch_norm=False, return_xvector=False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = MyLayer(input_dim=self.input_dim,output_dim=512,context_dim=5,dilation=1)
        self.fc2 = MyLayer(input_dim=512  ,output_dim=1536 ,context_dim=3, dilation=2)
        self.fc3 = MyLayer(input_dim=1536 ,output_dim=512  ,context_dim=3, dilation=3)
        self.fc4 = MyLayer(input_dim=512  ,output_dim=512  ,context_dim=1, dilation=1)
        self.fc5 = MyLayer(input_dim=512  ,output_dim=1500 ,context_dim=1, dilation=1)

        self.fc6 = tf.keras.layers.Dense(512)
        self.fc7 = tf.keras.layers.Dense(512)
        self.output_layer = tf.keras.layers.Dense(self.output_dim)

        if batch_norm:
            self.reg1 = Reg(dropout_rate)
            self.reg2 = Reg(dropout_rate)
            self.reg3 = Reg(dropout_rate)
            self.reg4 = Reg(dropout_rate)
            self.reg5 = Reg(dropout_rate)
            self.reg6 = Reg(dropout_rate)
        else:
            self.reg1 = tf.keras.layers.Dropout(rate=dropout_rate)
            self.reg2 = tf.keras.layers.Dropout(rate=dropout_rate)
            self.reg3 = tf.keras.layers.Dropout(rate=dropout_rate)
            self.reg4 = tf.keras.layers.Dropout(rate=dropout_rate)
            self.reg5 = tf.keras.layers.Dropout(rate=dropout_rate)
            self.reg6 = tf.keras.layers.Dropout(rate=dropout_rate)



    def call(self,x,training=True, return_logits=True):
        with tf.name_scope("Extractor"):
            with tf.name_scope("Fc1"):
                x = tf.nn.relu(self.fc1(x))
                x = self.reg1(x, training=training)
            with tf.name_scope("Fc2"):
                x = tf.nn.relu(self.fc2(x))
                x = self.reg2(x, training=training)
            with tf.name_scope("Fc3"):
                x = tf.nn.relu(self.fc3(x))
                x = self.reg3(x, training=training)
            with tf.name_scope("Fc4"):
                x = tf.nn.relu(self.fc4(x))
                x = self.reg4(x, training=training)
            with tf.name_scope("Fc5"):
                x = tf.nn.relu(self.fc5(x))
                x = self.reg5(x, training=training)
            with tf.name_scope("StatsPool"):
                x = self.statpool(x)

            with tf.name_scope("Segment6"):
                x = self.fc6(x)
        
        with tf.name_scope("Classifier"):
            x = tf.nn.relu(x)
            x = self.reg6(x)
            x = tf.nn.relu(self.fc7(x))
            x = self.output_layer(x)
        if return_logits:
            return x 
        else:
            x = self.softmax(x)
            return x

    def statpool(self,x):
        mu = tf.math.reduce_mean(x,axis=1)
        sigma = tf.math.reduce_std(x,axis=1)
        return tf.concat([mu,sigma],1)
        
n_feats = 40
model = MyModel(n_feats,3)

loss_fn =  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(loss=loss_fn, optimizer="adam",metrics = ["accuracy"])

def batch(N=4):
    for i in range(100):
        yield np.random.normal(size=(N,n_feats,200)), np.random.randint(0,3,size=(N))

history = model.fit(x = batch(N=300),
                    validation_data= batch(N=300),
                   )

import numpy as np
import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int = None,
        output_dim: int = 512,
        context_dim: int = 5,
        stride: int = 1,
        dilation: int = 1,
        kernel_initializer='glorot_uniform'
    ):
        super(MyLayer, self).__init__()
        self.input_dim = input_dim
        self.conv = tf.keras.layers.Conv1D(filters=output_dim,
                                           kernel_size=context_dim,
                                           strides=stride,
                                           dilation_rate=dilation,
                                           kernel_initializer=kernel_initializer)

    def call(self, x):
        batch_size, num_frames, num_feats = x.shape
        # if self.input_dim:
        #     assert self.input_dim == num_feats
        return self.conv(x)

class Reg(tf.keras.layers.Layer):
    def __init__(self,dropout_rate=0.2):
        super(Reg,self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.do = tf.keras.layers.Dropout(rate=dropout_rate)
    def call(self,x,training=True):
        x = self.bn(x,training=training)
        return self.do(x,training=training)
        
class MyModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim,dropout_rate=0.2,batch_norm=False, return_xvector=False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = MyLayer(input_dim=self.input_dim,output_dim=512,context_dim=5,dilation=1)
        self.fc2 = MyLayer(input_dim=512  ,output_dim=1536 ,context_dim=3, dilation=2)
        self.fc3 = MyLayer(input_dim=1536 ,output_dim=512  ,context_dim=3, dilation=3)
        self.fc4 = MyLayer(input_dim=512  ,output_dim=512  ,context_dim=1, dilation=1)
        self.fc5 = MyLayer(input_dim=512  ,output_dim=1500 ,context_dim=1, dilation=1)

        self.fc6 = tf.keras.layers.Dense(512)
        self.fc7 = tf.keras.layers.Dense(512)
        self.output_layer = tf.keras.layers.Dense(self.output_dim)

        if batch_norm:
            self.reg1 = Reg(dropout_rate)
            self.reg2 = Reg(dropout_rate)
            self.reg3 = Reg(dropout_rate)
            self.reg4 = Reg(dropout_rate)
            self.reg5 = Reg(dropout_rate)
            self.reg6 = Reg(dropout_rate)
        else:
            self.reg1 = tf.keras.layers.Dropout(rate=dropout_rate)
            self.reg2 = tf.keras.layers.Dropout(rate=dropout_rate)
            self.reg3 = tf.keras.layers.Dropout(rate=dropout_rate)
            self.reg4 = tf.keras.layers.Dropout(rate=dropout_rate)
            self.reg5 = tf.keras.layers.Dropout(rate=dropout_rate)
            self.reg6 = tf.keras.layers.Dropout(rate=dropout_rate)



    def call(self,x,training=True, return_logits=True):
        with tf.name_scope("Extractor"):
            with tf.name_scope("Fc1"):
                x = tf.nn.relu(self.fc1(x))
                x = self.reg1(x, training=training)
            with tf.name_scope("Fc2"):
                x = tf.nn.relu(self.fc2(x))
                x = self.reg2(x, training=training)
            with tf.name_scope("Fc3"):
                x = tf.nn.relu(self.fc3(x))
                x = self.reg3(x, training=training)
            with tf.name_scope("Fc4"):
                x = tf.nn.relu(self.fc4(x))
                x = self.reg4(x, training=training)
            with tf.name_scope("Fc5"):
                x = tf.nn.relu(self.fc5(x))
                x = self.reg5(x, training=training)
            with tf.name_scope("StatsPool"):
                x = self.statpool(x)

            with tf.name_scope("Segment6"):
                x = self.fc6(x)
        
        with tf.name_scope("Classifier"):
            x = tf.nn.relu(x)
            x = self.reg6(x)
            x = tf.nn.relu(self.fc7(x))
            x = self.output_layer(x)
        if return_logits:
            return x 
        else:
            x = self.softmax(x)
            return x

    def statpool(self,x):
        mu = tf.math.reduce_mean(x,axis=1)
        sigma = tf.math.reduce_std(x,axis=1)
        return tf.concat([mu,sigma],1)
        
n_feats = 40
model = MyModel(n_feats,3)

loss_fn =  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(loss=loss_fn, optimizer="adam",metrics = ["accuracy"])

def batch(N=4):
    for i in range(100):
        yield np.random.normal(size=(N,n_feats,200)), np.random.randint(0,3,size=(N))

history = model.fit(x = batch(N=300),
                    validation_data= batch(N=300),
                   )