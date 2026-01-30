import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import psutil


def memory():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return memory_use

class TestLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        
        super(TestLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.s = input_shape

    @tf.function
    def call(self, inputs, training=None):
        
        X = tf.matmul(tf.transpose(inputs),inputs)
        
        return tf.linalg.expm(X)
    


S = tf.keras.Input(shape=(1,), name="sequence", dtype=tf.float64)
T = TestLayer()(S)
model = tf.keras.Model(inputs=S,outputs=T)
dataset = tf.data.Dataset.from_tensors(tf.constant(1)).repeat().map(lambda x: tf.ones([tf.random.uniform([1], minval=100, maxval=3000, dtype=tf.int32)[0],1],dtype=tf.float64))

memory_usage = []

i = 0
for data in dataset:
    s = model(data)
    
    memory_usage.append(memory())

    if i == 1000:
        break
        
    i = i+1
    
plt.figure()
plt.plot(memory_usage)
plt.xlabel("iterations")
plt.ylabel("memory usage")
plt.show()