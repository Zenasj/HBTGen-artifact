import random

# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import numpy as np
from tensorflow.keras.utils import plot_model

from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
tf.keras.backend.set_floatx('float64')
#%%###################################################################
# y(n) = func_g(x)y(n-1) + X\beta + \alpha
###############################################################
def func_g (x):
    v = np.sin(x)
    return v
########################################################
#%% deine the neutron for X\beta + alpha
############################################
class Linear(layers.Layer):

  def __init__(self, CityNum, CityFactorNum):   
    self.CityNum = CityNum    
    self.CityFactorNum = CityFactorNum
    super(Linear, self).__init__()
    
  def build(self, input_shape):
    self.beta  = self.add_weight(shape=(self.CityFactorNum, 1),  initializer='random_normal', trainable=True)
    self.alpha = self.add_weight(shape=(self.CityNum,),        initializer='random_normal', trainable=True)

  def call(self, X):
    v = tf.matmul( X[:,1:], self.beta)  + self.alpha[ tf.dtypes.cast( X[0,0], tf.int32 ) ]
    return v
#############################################33
#%% define rnn cell node
###############################################
class MinimalRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, CityNum, CityFactorNum, **kwargs):
        self.units = units
        self.state_size = 100
        self.CityNum = CityNum    
        self.CityFactorNum = CityFactorNum
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.beta  = self.add_weight(shape=(self.CityNum,),  initializer='random_normal', trainable=True)
        self.alpha = self.add_weight(shape=(self.CityFactorNum-1,), initializer='random_normal', trainable=True)
        
        self.dense_1  = layers.Dense(32, activation='tanh')
        self.dense_2  = layers.Dense(64, activation='tanh')
        self.dense_3  = layers.Dense(64, activation='tanh')
        self.dense_4  = layers.Dense(64, activation='tanh')
        self.dense_5= layers.Dense(1)

        
        self.Xbeta_Add_alpha = Linear(self.CityNum, self.CityFactorNum)
                
        self.built = True
    
    def get_initial_state(self, inputs, batch_size=None, dtype=None):
        initial_states = []
        initial_states = [tf.ones(1)]

        return tuple(initial_states)
        

    def call(self, inputs, states):
 
        X_input = inputs[0]
        U_input = inputs[1]
     
        s1 = states[0] 

        gU = self.dense_1(U_input)
        gU = self.dense_2(gU)
        gU = self.dense_3(gU)
        gU = self.dense_4(gU)
        gU = self.dense_5(gU)
        X = self.Xbeta_Add_alpha(U_input)

        gUZ  = layers.dot( [gU, s1], axes=1, name = 'dot')
        gUZX = layers.add( [gUZ, X], name = 'add')
    
        output = [gUZ, gUZX]
        new_state = [gUZX]
        
        return output, new_state

#########################################
#%% define model 
sample_num = 2000
time_step = 100
CityNum, CityFactorNum = 100, 1
UFactorNum = 1
#############################################
def rnn_model (CityNum, CityFactorNum, time_step):
    
    input_X = tf.keras.Input(shape=[time_step, CityFactorNum])
    input_U = tf.keras.Input(shape=[time_step, UFactorNum])
    
    
    
    cells = MinimalRNNCell(1, CityNum, CityFactorNum)
    
    #rnn = tf.keras.layers.RNN(cells, return_sequences=True)(input)
    #out = tf.keras.layers.Dense(units=1)(rnn)

    out = tf.keras.layers.RNN(cells, return_sequences=True)([input_X, input_U])
    out = tf.keras.layers.Dense(1)(out)
    
    
    
    model = tf.keras.Model(inputs=[input_X, input_U], outputs=out)
    plot_model(model, to_file='model.png')
    
    model.summary()
    model.compile(optimizer='rmsprop',  loss=['mse'],  metrics=['mse'])
    
    return model 

################################################################
#%% set the test data
##(sample_num,  time_step_num, units)
######################################################################
sample_num = 2000
time_step = 100
CityNum, CityFactorNum = 100, 1

X = np.random.uniform(-10,10, size=(sample_num,time_step,1))
Y = np.zeros((sample_num,time_step,1))

#for i1 in range()


for i1 in range(sample_num):
    for i2 in range(1,time_step):
        for i3 in range(1):
            Y[i1, i2, i3] = func_g(X[i1, i2, i3]) +  Y[i1, i2-1, i3]

#-----------------------------------------------------------------------
model = rnn_model (CityNum, CityFactorNum, time_step)
model.fit(X, Y, batch_size = 100, epochs = 50)
#----------------------------------------------------------------------
start = np.random.uniform(-10,10, size=(1,time_step,1))
start = np.linspace(-10,10,time_step)
start = np.reshape(start, (1,time_step,1))
next = model.predict(start)


plt.plot(start[0,:,0], next[0,:,0],'rs')
################################################################
#%% true solution 
##############################################################
next = next
for i1 in range(1):
    for i2 in range(1,time_step):
        for i3 in range(1):
            next[i1, i2, i3] = func_g(start[i1, i2, i3]) + next[i1, i2-1, i3]

plt.plot(start[0,:,0], next[0,:,0],'bo')