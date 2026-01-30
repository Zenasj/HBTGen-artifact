import random

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from pprint import pprint
import time


network_architecture = {
    "channels" : 10,  # Size of z variables.
    "num_layers" : 6,  # Number of resnet blocks for each downsampling layer.
}

def initialize_conv2dwn_vars(x, kernel_shape, output_channels, stride, padding, init_scale=1.0, mask=None):

    input_shape = x.get_shape()
    filter_shape = [kernel_shape[0], kernel_shape[1], int(input_shape[-1]), output_channels]
    stride_shape = [1, stride[0], stride[1], 1]

    v_inizializer = tf.random_normal_initializer(0, 0.05)
    v = tf.get_variable("v", filter_shape, tf.float32, v_inizializer)
#     see https://www.tensorflow.org/api_docs/python/tf/Variable#initialized_value
    v_aux = v.initialized_value()
    
    if mask is not None:  # used for auto-regressive convolutions.
        v_aux = mask * v_masked
    
    v_norm = tf.nn.l2_normalize(v_aux, [0, 1, 2])
    x_init = tf.nn.conv2d(x, v_norm, strides=stride_shape, padding=padding) # ***
    m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
    scale_init = init_scale / tf.sqrt(v_init + 1e-10)

    h_aux = tf.reshape(scale_init, [1, 1, 1, -1]) * (x_init - tf.reshape(m_init, [1, 1, 1, -1]))

    g = tf.get_variable("g", initializer=tf.log(scale_init) / 3.0)
    b = tf.get_variable("b", initializer=-m_init * scale_init)
            
    return h_aux

def initializers_for_conv2dwn_vars(x, kernel_shape, output_channels, stride, padding, init_scale=1.0, mask=None):

    input_shape = x.get_shape()
    filter_shape = [kernel_shape[0], kernel_shape[1], int(input_shape[-1]), output_channels]
    stride_shape = [1, stride[0], stride[1], 1]
    
    v_aux = tf.constant(np.random.normal(loc=0, scale=0.05, size=filter_shape), dtype=tf.float32, name="v_aux")

    if mask is not None:  # used for auto-regressive convolutions.
        v_aux = mask * v_masked
    
    v_norm = tf.nn.l2_normalize(v_aux, [0, 1, 2])
    x_init = tf.nn.conv2d(x, v_norm, strides=stride_shape, padding=padding) # ***
    m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
    scale_init = init_scale / tf.sqrt(v_init + 1e-10)

    def g_inizializer(*args, **kwargs):
        return tf.log(scale_init) / 3.0
    
    def b_inizializer(*args, **kwargs):
        return -m_init * scale_init

    def v_inizializer(*args, **kwargs):
        return v_aux
    
    h_aux = tf.reshape(scale_init, [1, 1, 1, -1]) * (x_init - tf.reshape(m_init, [1, 1, 1, -1]))
            
    return {'v' : v_inizializer, 'g' : g_inizializer, 'b' : b_inizializer}, h_aux


def conv2dwn_reuse_vars(inputs, kernel_shape, output_channels, stride, padding, mask):

    input_shape = inputs.get_shape()
    filter_shape = [kernel_shape[0], kernel_shape[1], int(input_shape[-1]), output_channels]
    stride_shape = [1, stride[0], stride[1], 1]
    print("...ready v")
    v = tf.get_variable("v", shape=filter_shape)
    print("...ready g")
    g = tf.get_variable("g", shape=[output_channels]) #initializer=initializers['g'],
    print("...ready b")
    b = tf.get_variable("b", shape=[output_channels]) # initializer=initializers['b'],
    print("...done vars")
    if mask is not None:
        v = mask * v

    # use weight normalization (Salimans & Kingma, 2016)
    w = tf.reshape(tf.exp(g), [1, 1, 1, output_channels]) * tf.nn.l2_normalize(v, [0, 1, 2])

    # calculate convolutional layer output
    b = tf.reshape(b, [1, 1, 1, -1])
    
    print("...ready")
    r = tf.nn.conv2d(inputs, w, stride_shape, padding) + b
    print("...done")
        
    return r

def conv2dwn_create_vars(inputs, initializers, kernel_shape, output_channels, stride, padding, mask):

    input_shape = inputs.get_shape()
    filter_shape = [kernel_shape[0], kernel_shape[1], int(input_shape[-1]), output_channels]
    stride_shape = [1, stride[0], stride[1], 1]
    print("...ready v")
    v = tf.get_variable("v", shape=filter_shape, initializer=initializers['v'])
    print("...ready g")
    g = tf.get_variable("g", shape=[output_channels], initializer=initializers['g'])
    print("...ready b")
    b = tf.get_variable("b", shape=[output_channels], initializer=initializers['b'])
    print("...done vars")
    if mask is not None:
        v = mask * v

    # use weight normalization (Salimans & Kingma, 2016)
    w = tf.reshape(tf.exp(g), [1, 1, 1, output_channels]) * tf.nn.l2_normalize(v, [0, 1, 2])

    # calculate convolutional layer output
    b = tf.reshape(b, [1, 1, 1, -1])
    
    print("...ready")
    r = tf.nn.conv2d(inputs, w, stride_shape, padding) + b
    print("...done")
        
    return r


# In[8]:


# REUSE VARIABLES

def conv2d_weightnorm_layer(name, inputs, inputs_aux, n_channels, kernel_shape=(3,3), stride=(1,1), init_scale=1.0, mask=None):
    
    conv2dwn_kwargs = {"kernel_shape" : kernel_shape,
                       "stride" : stride,
                       "padding" : 'SAME',
                       "mask" : mask
    }

    print("creating layer " + name)
    
    with tf.variable_scope(name, reuse=None):#, reuse=tf.AUTO_REUSE):
        h_aux = initialize_conv2dwn_vars(inputs_aux,
                                      output_channels = n_channels,
                                      init_scale = init_scale,
                                      **conv2dwn_kwargs)

    print("middle")

    with tf.variable_scope(name, reuse=True):#, reuse=tf.AUTO_REUSE):
        h = conv2dwn_reuse_vars(inputs, output_channels = n_channels, **conv2dwn_kwargs)
        
    print("done")
    print(h, h_aux)
        
    return h, h_aux


class Network:
    
    def __init__(self, network_architecture):
        self._num_layers = network_architecture["num_layers"]
        self._channels = network_architecture["channels"]
    
    def _build_net(self, x):
        
        h, h_aux = conv2d_weightnorm_layer("first_layer_conv",
                                        x,
                                        x,
                                        self._channels,
                                        kernel_shape = (5,5),
                                        stride = (2,2)
                                       )

        print("start loop")
        for i in range(self._num_layers):
            print("\n layer %d"%i)
            h, h_aux = conv2d_weightnorm_layer("layer%d"%i,
                                    h,
                                    h_aux,
                                    self._channels,
                                    kernel_shape = (5,5),
                                    stride = (1,1)
                                   )
            print("DONE layer %d \n"%i)
                
        return h, h_aux
    


# In[9]:

tf.reset_default_graph()

print("\n\nGRAPH CREATION WITH WEIGHT SHARING...\n\n")
t_i = time.time()

x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
net = Network(network_architecture)
output = net._build_net(x)

t_f = time.time()
print("\nEND OF GRAPH CREATION WITH WEIGHT SHARING")
print("time : %g s\n\n"%(t_f - t_i))

# In[ ]:


g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
pprint([str(var.name)+" "+str(var.get_shape().as_list()) for var in g_vars])
print(len(g_vars))


# In[ ]:


# PASS INITIALIZERS

def conv2d_weightnorm_layer(name, inputs, inputs_aux, n_channels, kernel_shape=(3,3), stride=(1,1), init_scale=1.0, mask=None):
    
    conv2dwn_kwargs = {"kernel_shape" : kernel_shape,
                       "stride" : stride,
                       "padding" : 'SAME',
                       "mask" : mask
    }

    print("creating layer " + name)
    
    with tf.variable_scope(name, reuse=None):#, reuse=tf.AUTO_REUSE):
        initializers, h_aux = initializers_for_conv2dwn_vars(inputs_aux,
                                                  output_channels = n_channels,
                                                  init_scale = init_scale,
                                                  **conv2dwn_kwargs)
    
    print("middle")

    with tf.variable_scope(name, reuse=None):#, reuse=tf.AUTO_REUSE):
        h = conv2dwn_create_vars(inputs,
                                initializers = initializers,
                                output_channels = n_channels,
                                **conv2dwn_kwargs)
        
    print("done")
    print(h, h_aux)
    
    return h, h_aux


class Network:
    
    def __init__(self, network_architecture):
        self._num_layers = network_architecture["num_layers"]
        self._channels = network_architecture["channels"]
    
    def _build_net(self, x):
        
        h, h_aux = conv2d_weightnorm_layer("first_layer_conv",
                                        x,
                                        x,
                                        self._channels,
                                        kernel_shape = (5,5),
                                        stride = (2,2)
                                       )

        print("start loop")
        for i in range(self._num_layers):
            print("\n layer %d"%i)
            h, h_aux = conv2d_weightnorm_layer("layer%d"%i,
                                    h,
                                    h_aux,
                                    self._channels,
                                    kernel_shape = (5,5),
                                    stride = (1,1)
                                   )
            print("DONE layer %d \n"%i)
                
        return h, h_aux
    


# In[ ]:


tf.reset_default_graph()

print("\n\nGRAPH CREATION WITH INITIALIZERS FROM OTHER TENSORS...\n\n")
t_i = time.time()

x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
net = Network(network_architecture)
output = net._build_net(x)

t_f = time.time()
print("\nEND OF GRAPH CREATION WITH INITIALIZERS FROM OTHER TENSORS")
print("time : %g s\n\n"%(t_f - t_i))

# In[ ]:
g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
pprint([str(var.name)+" "+str(var.get_shape().as_list()) for var in g_vars])
print(len(g_vars))