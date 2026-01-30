import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
    
    def build_graph(self, input_shape): 
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        
        _ = self.call(inputs)
        

model = MyModel()
model.build_graph((32,10,))

model.summary()

input = np.random.random((32,10,))
input = input.astype(np.float32)
output = model.call(input)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = output.eval(session=sess)

print('Model output shape:')
print(result.shape)