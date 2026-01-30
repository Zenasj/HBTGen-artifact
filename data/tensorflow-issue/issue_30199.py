import math
import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from keras.layers import *
from keras.models import Model
import numpy as np

# I define the representation model
inp_dim = 5
outp_dim = 5
inp = Input(shape=( inp_dim,), name="input")
outp = Dense(outp_dim, activation='tanh', name="simple_layer")(inp)
model = Model(inputs=inp, outputs=outp) 

# I define over here part of my overal loss (note that defining the full loss doesn't change the output.)
anchor = tf.placeholder(tf.float32, [None, inp_dim], name="representation_shift")
positive = tf.placeholder(tf.float32, [None, inp_dim], name="representation_shift")
outp_anchor = model(anchor)
outp_positive = model(positive)
u = tf.math.subtract(outp_anchor, outp_positive)
u_norm = tf.linalg.norm(u, axis=-1) #u_norm is the loss am trying to minimize in this example

#I define here my gradient operations
_OPTIMIZER = tf.train.AdamOptimizer()
grads_and_vars = _OPTIMIZER.compute_gradients(u_norm)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

# I generate a dummy example
batch_size = 10
random_input1 = np.random.rand(batch_size, inp_dim)
random_input2 = np.random.rand(batch_size, inp_dim)
# And I force it to the case where parts of the data is similar (juste the first line actually)
random_input2[0] = random_input1[0] 

def replace_none_with_zero(l):
  return [0 if i==None else i for i in l]
grads= replace_none_with_zero(grads_and_vars)

sess.run(grads, feed_dict={
    anchor: random_input1,
    positive: random_input2, 
})[0][0]
# This outputs "nan" erroneously