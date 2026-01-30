import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

def make_model():
    inp = tf.keras.Input(1)
    y = tf.keras.layers.Dense(1)(inp)
    y_times_2 = tf.math.multiply(2., y)
    model = tf.keras.Model(inputs=inp, outputs={'y':y, 'y_times_2':y_times_2})
    return model

import tensorflow as tf
import numpy as np

print('Using Tensorflow version {} (git version {})'.format(tf.version.VERSION, tf.version.GIT_VERSION))

x = np.random.normal(size=(64,))
y = x

def make_model():
    inp = tf.keras.Input(1)
    y = tf.keras.layers.Dense(1)(inp)
    y_times_2 = tf.math.multiply(2., y)
    model = tf.keras.Model(inputs=inp, outputs={'y':y, 'y_times_2':y_times_2})
    return model

model = make_model()

model.summary()

try:
    print('Trying compilation with the output names')
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1), 
                  loss={'y':'mean_squared_error', 'y_times_2':'mean_squared_error'})
except Exception as e:
    print('The compilation failed with the following message error:')
    print(e)
    print('Trying compilation with the layers names')
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
                  loss={'dense':'mean_squared_error', 'tf_op_layer_Mul':'mean_squared_error'})
    
try:
    print('\nTrying training with the output names')
    model.fit(x, {'y':y, 'y_times_2':2*y}, epochs=2)
except Exception as e:
    print('The training failed with the following message error:')
    print(e)
    print('Trying training with the layers names')
    model.fit(x, {'dense':y, 'tf_op_layer_Mul':2*y}, epochs=2)
    
print('\n###################\nAdd metrics to the example:')

model = make_model()

model.summary()

try:
    print('Trying compilation with the output names')
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1), 
                  loss={'dense_1':'mean_squared_error', 'tf_op_layer_Mul_1':'mean_squared_error'},
                  metrics={'y':'mean_absolute_error', 'y_times_2':'mean_absolute_error'})
except Exception as e:
    print('The compilation failed with the following message error:')
    print(e)
    print('Trying compilation with the layers names')
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
                  loss={'dense_1':'mean_squared_error', 'tf_op_layer_Mul_1':'mean_squared_error'},
                  metrics={'dense_1':'mean_absolute_error', 'tf_op_layer_Mul_1':'mean_absolute_error'})

model.fit(x, {'dense_1':y, 'tf_op_layer_Mul_1':2*y}, epochs=2)