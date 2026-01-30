import random
from tensorflow import keras
from tensorflow.keras import layers

mymodel.fit(x = {'input_0' : train_data_type_0, 'input_1':  train_data_type_1}, y = train_labels, 
         validation_data = ({'input_0' : val_data_type_0, 'input_1':  val_data_type_1}, val_labels) )

import numpy as np
import tensorflow as tf

print(tf.__version__)

train_input_0 = np.random.rand(1000, 1)
train_input_1 = np.random.rand(1000, 1)
train_labels  = np.random.rand(1000, 1)

val_input_0 = np.random.rand(1000, 1)
val_input_1 = np.random.rand(1000, 1)
val_labels  = np.random.rand(1000, 1)

input_0 = tf.keras.Input(shape=(None,), name='input_0')
input_1 = tf.keras.Input(shape=(None,), name='input_1')

class my_model(tf.keras.Model):
    def __init__(self):
        super(my_model, self).__init__(self)
        self.hidden_layer_0 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.hidden_layer_1 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.concat         = tf.keras.layers.Concatenate()

        self.out_layer    = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs =  [input_0, input_1]):
        activation_0 = self.hidden_layer_0(inputs['input_0'])
        activation_1 = self.hidden_layer_1(inputs['input_1'])
        concat       = self.concat([activation_0, activation_1])
      
        return self.out_layer(concat)

model = my_model()
opt = tf.optimizers.Adam()
loss = tf.keras.losses.MeanAbsoluteError()
model.compile(opt, loss)

model.fit(x = {'input_0' : train_input_0, 'input_1':  train_input_1}, y = train_labels, 
         validation_data = ({'input_0' : val_input_0, 'input_1':  val_input_1}, val_labels) )