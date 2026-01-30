import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dense(inputs):
    return tf.layers.dense(inputs,units=513,activation='relu')

x_mixed=tf.keras.Input(shape=(5, 256),dtype=tf.float32, name='input')
fc_out=tf.keras.layers.Lambda(dense)(x_mixed)

model=tf.keras.Model(inputs=x_mixed,outputs=fc_out)
model.compile(loss="mse",metrics=['mae'],optimizer='adam')
model.summary()

X=np.random.randn(1000,5,256)
y=np.random.randn(1000,5,513)
model.fit(X,y,batch_size=16,epochs=5)