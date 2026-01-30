from tensorflow.keras import optimizers

import time
import math
import tensorflow as tf
import numpy as np
import tensorflow as keras
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from scipy . stats import multivariate_normal as normal

d = 10
T = 0.1
n_time = 5
n_sample = 100
batch_size = 100
n_maxstep = 400
h = (T + 0.0) / n_time
t_stamp = np.arange (0, n_time) * h

def f_tf (t, X, Y, Z):
    V =  Y - tf.math.sin (Y)
    return V

def g_tf (t, X):
    V =  tf.math.reduce_sum (X**3, 1,keepdims=True)
    return V

def k_tf ( n_sample ):
    W = np.zeros ([ n_sample, d, n_time  ], dtype = np.float64)
    X_sample  = np.zeros ([ n_sample, d, n_time+1], dtype = np.float64)
    for i in range (n_time):
        W [:, :, i  ] = np.reshape ( normal.rvs ( mean = np.zeros(d,dtype = np.float64),\
                                      cov =1, size = n_sample ), ( n_sample, d))
        X_sample  [:, :, i+1] =  W [:, :, i]
    return W, X_sample

def nn_tf(x):
    x = keras.layers.BatchNormalization(batch_size = n_sample)(x)
    x = keras.layers.Dense(d, batch_size = n_sample)(x)
    x = keras.layers.BatchNormalization(batch_size = n_sample)(x)
    return x

dW = keras.Input(shape = (d, n_time  ), batch_size=n_sample, dtype = tf.float64, name = 'dW')
XX = keras.Input(shape = (d, n_time+1), batch_size=n_sample, dtype = tf.float64, name = 'X' )
X = XX
Y = tf.zeros([n_sample, 1], dtype = tf.float64)
Z = tf.zeros([n_sample, d], dtype = tf.float64)

for it in range(n_time-1):
    with tf.name_scope(str(it+1)):
        Y = Y +  tf.math.reduce_sum( Z * dW[:,:,it],  1, keepdims=True)
        subX = tf.reshape(X[:,:,it], shape = [n_sample, d])
        Z = nn_tf(subX) / d

Y = Y + tf.math.reduce_sum (Z * dW [:, :, n_time-1], 1, keepdims=True)
model = keras.Model(inputs=[XX,dW], outputs=[Y])

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
dW_train, X_train = k_tf ( n_sample )

for epoch in range (10):
    with tf.GradientTape() as tape:
        predictions = model( [X_train, dW_train] )
        label = g_tf (T, X_train[:, :, n_time])
        loss_value = tf.reduce_sum( tf.keras.losses.MSE (label, predictions ) )
    grads = tape.gradient(loss_value,  model.trainable_variables)
    optimizer.apply_gradients( zip(grads, model.trainable_variables) )
    accuracy = train_accuracy(label, predictions)
    print("step ", epoch, "loss = ", loss_value.numpy(), "accuracy = ", accuracy.numpy())

import time
import math
import tensorflow as tf
import numpy as np
import tensorflow as keras
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from scipy . stats import multivariate_normal as normal
# tf.keras.backend.set_floatx('float64') # sets dtype as tf.float64

d = 10
T = 0.1
n_time = 5
n_sample = 100
batch_size = 100
n_maxstep = 400
h = (T + 0.0) / n_time
t_stamp = np.arange (0, n_time) * h

def f_tf (t, X, Y, Z):
    V =  Y - tf.math.sin (Y)
    return V

def g_tf (t, X):
    V =  tf.math.reduce_sum (X**3, 1,keepdims=True)
    return V

def k_tf ( n_sample ):
    W = np.zeros ([ n_sample, d, n_time  ], dtype = np.float64)
    X_sample  = np.zeros ([ n_sample, d, n_time+1], dtype = np.float64)
    for i in range (n_time):
        W [:, :, i  ] = np.reshape ( normal.rvs ( mean = np.zeros(d,dtype = np.float64),\
                                      cov =1, size = n_sample ), ( n_sample, d))
        X_sample  [:, :, i+1] =  W [:, :, i]
    return W, X_sample

def nn_tf(x):
    x = keras.layers.BatchNormalization(batch_size = n_sample,dtype=tf.float64)(x)
    x = keras.layers.Dense(d, batch_size = n_sample)(x)
    x = keras.layers.BatchNormalization(batch_size = n_sample,dtype=tf.float64)(x)
    return x

dW = keras.Input(shape = (d, n_time  ), batch_size=n_sample, dtype = tf.float64, name = 'dW')
XX = keras.Input(shape = (d, n_time+1), batch_size=n_sample, dtype = tf.float64, name = 'X' )
X = XX
Y = tf.zeros([n_sample, 1], dtype = tf.float64)
Z = tf.zeros([n_sample, d], dtype = tf.float64)

for it in range(n_time-1):
    with tf.name_scope(str(it+1)):
        Y = Y +  tf.math.reduce_sum( Z * dW[:,:,it],  1, keepdims=True)
        subX = tf.reshape(X[:,:,it], shape = [n_sample, d])
        Z = nn_tf(subX) / d

Y = Y + tf.math.reduce_sum (Z * dW [:, :, n_time-1], 1, keepdims=True)
model = keras.Model(inputs=[XX,dW], outputs=[Y])

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
dW_train, X_train = k_tf ( n_sample )

for epoch in range (10):
    with tf.GradientTape() as tape:
        predictions = model( [X_train, dW_train] )
        label = g_tf (T, X_train[:, :, n_time])
        loss_value = tf.reduce_sum( tf.keras.losses.MSE (label, predictions ) )
    grads = tape.gradient(loss_value,  model.trainable_variables)
    optimizer.apply_gradients( zip(grads, model.trainable_variables) )
    accuracy = train_accuracy(label, predictions)
    print("step ", epoch, "loss = ", loss_value.numpy(), "accuracy = ", accuracy.numpy())