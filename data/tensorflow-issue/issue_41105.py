import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

def rate_mse(rate=1e5):
    @tf.function # also needed for printing
    def loss(y_true, y_pred):
        tmp = rate*K.mean(K.square(y_pred - y_true), axis=-1)
#        tf.print('shape %s and rank %s output in mse'%(K.shape(tmp), tf.rank(tmp)))
        tf.print('shape and rank output in mse',[K.shape(tmp), tf.rank(tmp)])
        tf.print('mse loss:',tmp) # print when I put tf.function
        return tmp
    return loss

class newLayer(tf.keras.layers.Layer):
    def __init__(self, rate=5e-2, **kwargs):
        super(newLayer, self).__init__(**kwargs)
        self.rate = rate
        
#    @tf.function # to be commented for NN training
    def call(self, inputs):
        tmp = self.rate*K.mean(inputs*inputs, axis=-1)
        tf.print('shape and rank output in regularizer',[K.shape(tmp), tf.rank(tmp)])
        tf.print('regularizer loss:',tmp)
        self.add_loss(tmp, inputs=True)
        return inputs

tot_n = 10000
xx = np.random.rand(tot_n,1)
yy = np.pi*xx

train_size = int(0.9*tot_n)
xx_train = xx[:train_size]; xx_val = xx[train_size:]
yy_train = yy[:train_size]; yy_val = yy[train_size:]

reg_layer = newLayer()

input_layer = Input(shape=(1,))                                      # input
hidden = Dense(20, activation='relu', input_shape=(2,))(input_layer) # hidden layer
hidden = reg_layer(hidden)
output_layer = Dense(1, activation='linear')(hidden) 

model = Model(inputs=[input_layer], outputs=[output_layer])
model.compile(optimizer='Adam', loss=rate_mse(), experimental_run_tf_function=False)
#model.compile(optimizer='Adam', loss=None, experimental_run_tf_function=False)
history = model.fit(xx_train, yy_train, epochs=50, batch_size = 100, 
                    validation_data=(xx_val,yy_val), verbose=2)

print(model.predict(np.array([[1]]))) # sanity check

K.mean(inputs * inputs, axis=-1)

K.mean(inputs * inputs)