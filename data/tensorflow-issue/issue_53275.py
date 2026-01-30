import math
import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

def complex_uniform_initializer(scale=0.05):
    real_initializer = tf.keras.initializers.RandomUniform(-scale,scale)
    def initializer(shape,dtype):
        if dtype == tf.complex64:
            dtype = tf.float32
        elif dtype == tf.complex128:
            dtype = tf.float64
        real = real_initializer(shape,dtype)
        imag = real_initializer(shape,dtype)
        return tf.dtypes.complex(real,imag)
    return initializer

class ComplexDenseLayer(tf.keras.layers.Layer):

    def __init__(self, out_units, activation=None):
        super().__init__()
        self.out_units = out_units
        self.activation = activation

    def build(self, input_shape):
        inp_units = input_shape[-1]
        initializer = complex_uniform_initializer()
        self.w = self.add_weight(shape=[inp_units, self.out_units],
                                 initializer = initializer,
                                 dtype=tf.complex64, trainable=True)
        self.b = self.add_weight(shape=[self.out_units],
                                 initializer = initializer,
                                 dtype=tf.complex64, trainable=True)

    def call(self,inp):
        x = tf.einsum('bi,ij->bj', inp, self.w)
        x = tf.nn.bias_add(x, self.b)
        return self.activation(x)

    

def model(input_units, intermediate_units, output_units):
    inp = tf.keras.layers.Input((input_units,))
    xreal = tf.keras.layers.Dense(intermediate_units)(inp)
    ximag = tf.keras.layers.Dense(intermediate_units)(inp)
    x = tf.cast(xreal, 'complex64') + 1j*tf.cast(ximag,'complex64')
    x = ComplexDenseLayer(intermediate_units, activation = lambda w: w * tf.math.conj(w))(x)
    x = tf.math.real(x)
    x = tf.keras.layers.Dense(output_units)(x)
    return tf.keras.Model(inp,x) 

nsamples = 100
bsize = 10
ninp,nintermediate,nout = 16,128,16
inp = np.random.rand(nsamples, ninp)
tar = np.random.rand(nsamples, nout)
data = tf.data.Dataset.from_tensor_slices((inp,tar)).batch(bsize)

#Single GPU training works fine
model1 = model(ninp,nintermediate,nout)
model1.summary()
model1.compile(loss='mse', optimizer='adam')
model1.fit(data)

#Distributed training fails
distribute_strategy =  tf.distribute.MirroredStrategy()
with distribute_strategy.scope():
    model2 = model(ninp,nintermediate,nout)
    model2.summary()
    model2.compile(loss='mse', optimizer='adam')
    model2.fit(data)