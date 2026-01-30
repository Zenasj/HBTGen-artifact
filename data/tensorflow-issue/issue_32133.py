from tensorflow import keras
from tensorflow.keras import layers

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs
    def build(self, input_shape):
        self.kernel = self.add_variable("kernel", initializer=tf.keras.initializers.GlorotUniform(),
                                        shape=[int(input_shape[-1]),
                                               self.num_outputs])
        self.bias = self.add_variable("bias", initializer=tf.zeros_initializer,
                                        shape=[self.num_outputs])
    def call(self, input):
        return tf.matmul(input, self.kernel) + self.bias

net = tf.keras.Sequential()
net.add(MyDenseLayer(100))
net.add(tf.keras.layers.ReLU())
net.add(MyDenseLayer(100))
net.add(tf.keras.layers.ReLU())
net.add(MyDenseLayer(1))
net.build((None, input_dim))

import os
import pickle
import random
import numpy as np
import tensorflow as tf

os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'

seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# NN Model
input_dim = 5
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(100, activation=tf.nn.relu, kernel_initializer=None))
net.add(tf.keras.layers.Dense(100, activation=tf.nn.relu, kernel_initializer=None))
net.add(tf.keras.layers.Dense(1, activation=None, kernel_initializer=None))
net.build((None, input_dim))

# Initial v_params
initial_v_params = net.variables

# Update NN Model one-step
x = np.random.normal(loc=0, scale=1., size=[1000, input_dim])
y = np.random.normal(loc=0, scale=1., size=[1000])

with tf.GradientTape() as tape:
    loss = tf.reduce_mean(tf.square(y - net(x)))
grad = tape.gradient(loss, net.trainable_variables)

# Tag for comparing files
tag = 1

with open('./numpy_data{}.pkl'.format(tag), 'wb') as f:
    pickle.dump([x, y], f)

with open('./initial_params{}.pkl'.format(tag), 'wb') as f:
    pickle.dump(initial_v_params, f)

with open('./loss{}.pkl'.format(tag), 'wb') as f:
    pickle.dump(loss, f)

with open('./grad{}.pkl'.format(tag), 'wb') as f:
    pickle.dump(grad, f)