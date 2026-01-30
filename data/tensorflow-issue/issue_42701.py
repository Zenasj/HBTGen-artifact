import random
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras

layers = tf.keras.layers

strategy = tf.distribute.MirroredStrategy()

class Test(layers.Layer):

    def __init__(self, memory, **kargs):
        super(Test, self).__init__(**kargs)
        
        self.memory = memory
        self.dense = layers.Dense(128)

    def call(self, inputs):
        
        res = tf.matmul(inputs, self.memory, transpose_b=True)
        res = self.dense(res)

        return res
    
def create_model(memory):
    
    input_tensor = layers.Input(
        shape=[128], name="input_tensor"
    )
 
    x = layers.Dense(128)(input_tensor)
    
    output = Test(memory)(x)

    return keras.Model(inputs=input_tensor, outputs=output)
    
with tf.device('/cpu'):
    
    memory = tf.Variable(tf.random.uniform([100, 128]), trainable=False)
    
    
with strategy.scope():
    
    model = create_model(memory)
    
model.compile()

model.summary()

print(memory)

import tensorflow as tf
from tensorflow import keras

layers = tf.keras.layers

strategy = tf.distribute.MirroredStrategy()

class Test(layers.Layer):

    def __init__(self, memory, **kargs):
        super(Test, self).__init__(**kargs)
        
        self.memory = memory
        self.dense = layers.Dense(128)

    def call(self, inputs):
        
        res = tf.matmul(inputs, self.memory, transpose_b=True)
        res = self.dense(res)

        return res
    
def create_model(memory):
    
    input_tensor = layers.Input(
        shape=[128], name="input_tensor"
    )
 
    x = layers.Dense(128)(input_tensor)
    
    output = Test(memory)(x)

    return keras.Model(inputs=input_tensor, outputs=output)
    

with strategy.scope():
    
    with tf.device('/cpu'):
    
        memory = tf.Variable(tf.random.uniform([100, 128]), trainable=False)
    
    model = create_model(memory)
    
model.compile()

model.summary()

print(memory)