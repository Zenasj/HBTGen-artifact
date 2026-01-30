import math
import random
from tensorflow import keras
from tensorflow.keras import layers

#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

# please ignore this function, it is just for generating a sparse tensor
def Dense2Sparse():
  dense = tf.keras.Input((None, None, None)); # dense.shape = (batch, num_heads, query_length, key_length)
  mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, query_length or 1, key_length)
  reshaped_mask = tf.keras.layers.Lambda(lambda x: tf.cond(tf.math.not_equal(tf.shape(x[0])[2], tf.shape(x[1])[2]), lambda: tf.tile(x[0], [1,1,tf.shape(x[1])[2],1,]), lambda: x[0]))([mask, dense]); # mask.shape = (batch, 1, query_length, key_length)
  reshaped_mask = tf.keras.layers.Lambda(lambda x: tf.tile(x[0],[1,tf.shape(x[1])[1],1,1]))([reshaped_mask, dense]); # mask.shape = (batch, num_heads, query_length, key_length)
  indices = tf.keras.layers.Lambda(lambda x: tf.where(tf.cast(x, dtype = tf.int32)))(reshaped_mask); # indices.shape = (num non zero values, 4)
  values = tf.keras.layers.Lambda(lambda x: tf.gather_nd(x[0], x[1]))([dense, indices]); # values.shape = (num non zero values)
  sparse = tf.keras.layers.Lambda(lambda x: tf.sparse.SparseTensor(x[0], values = x[1], dense_shape = tf.cast(tf.shape(x[2]), dtype = tf.int64)))([indices, values, dense]);
  return tf.keras.Model(inputs = (dense, mask), outputs = sparse);

# NOTE: ***this layer cannot be executed in graph mode***
class SparseDenseMatMul(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(SparseDenseMatMul, self).__init__(**kwargs);
  def call(self, inputs):
    a = inputs[0]; # a.shape = (batch, heads, query_length, key_length)
    b = inputs[1]; # b.shape = (batch, heads, key_length, value_dim)
    reshaped_a = tf.sparse.reshape(a, (-1, tf.shape(a)[-2], tf.shape(a)[-1])); # reshaped_a.shape = (batch * heads, query_length, key_length)
    reshaped_b = tf.reshape(b, (-1, tf.shape(b)[-2], tf.shape(b)[-1])); # reshaped_b.shape = (batch * heads, key_length, value_dim)
    def dot(x):
      a = x[0];
      b = x[1];
      c = tf.sparse.sparse_dense_matmul(a,b);
      return c; # c.shape = (query_length, value_dim)
    results = tf.map_fn(dot, (reshaped_a, reshaped_b), fn_output_signature = tf.TensorSpec((tf.shape(reshaped_a)[-2], tf.shape(reshaped_b)[-1]), dtype = tf.float32));
    results = tf.reshape(results, (tf.shape(a)[0], tf.shape(a)[1], tf.shape(results)[-2], tf.shape(results)[-1]));
    return results;

a = np.random.normal(size = (4,3,10,40));
mask = np.random.randint(low = 0, high = 2, size = (4,1,10,40));
a = Dense2Sparse()([a, mask]);
b = np.random.normal(size = (4,3,40,20)).astype(np.float32);

print(SparseDenseMatMul()([a,b])); # eager mode is OK

inputs_a = tf.keras.Input((None, None, None), sparse = True);
inputs_b = tf.keras.Input((None, None, None));
results_c = SparseDenseMatMul()([inputs_a,inputs_b]);
model = tf.keras.Model(inputs = (inputs_a, inputs_b), outputs = results_c);
print(model([a,b])); # graph mode is failed

#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

# please ignore this function, it is just for generating a sparse tensor
def Dense2Sparse():
  dense = tf.keras.Input((None, None, None)); # dense.shape = (batch, num_heads, query_length, key_length)
  mask = tf.keras.Input((1, None, None)); # mask.shape = (batch, 1, query_length or 1, key_length)
  reshaped_mask = tf.keras.layers.Lambda(lambda x: tf.cond(tf.math.not_equal(tf.shape(x[0])[2], tf.shape(x[1])[2]), lambda: tf.tile(x[0], [1,1,tf.shape(x[1])[2],1,]), lambda: x[0]))([mask, dense]); # mask.shape = (batch, 1, query_length, key_length)
  reshaped_mask = tf.keras.layers.Lambda(lambda x: tf.tile(x[0],[1,tf.shape(x[1])[1],1,1]))([reshaped_mask, dense]); # mask.shape = (batch, num_heads, query_length, key_length)
  indices = tf.keras.layers.Lambda(lambda x: tf.where(tf.cast(x, dtype = tf.int32)))(reshaped_mask); # indices.shape = (num non zero values, 4)
  values = tf.keras.layers.Lambda(lambda x: tf.gather_nd(x[0], x[1]))([dense, indices]); # values.shape = (num non zero values)
  sparse = tf.keras.layers.Lambda(lambda x: tf.sparse.SparseTensor(x[0], values = x[1], dense_shape = tf.cast(tf.shape(x[2]), dtype = tf.int64)))([indices, values, dense]);
  return tf.keras.Model(inputs = (dense, mask), outputs = sparse);

# NOTE: ***this layer cannot be executed in graph mode***
class SparseDenseMatMul(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(SparseDenseMatMul, self).__init__(**kwargs);
  def call(self, inputs):
    a = inputs[0]; # a.shape = (batch, heads, query_length, key_length)
    b = inputs[1]; # b.shape = (batch, heads, key_length, value_dim)
    reshaped_a = tf.sparse.reshape(a, (-1, tf.shape(a)[-2], tf.shape(a)[-1])); # reshaped_a.shape = (batch * heads, query_length, key_length)
    reshaped_b = tf.reshape(b, (-1, tf.shape(b)[-2], tf.shape(b)[-1])); # reshaped_b.shape = (batch * heads, key_length, value_dim)
    def dot(x):
      a = x[0];
      b = x[1];
      c = tf.sparse.sparse_dense_matmul(a,b);
      return c; # c.shape = (query_length, value_dim)
    results = tf.map_fn(dot, (reshaped_a, reshaped_b), fn_output_signature = tf.TensorSpec((tf.shape(reshaped_a)[-2], tf.shape(reshaped_b)[-1]), dtype = tf.float32));
    results = tf.reshape(results, (tf.shape(a)[0], tf.shape(a)[1], tf.shape(results)[-2], tf.shape(results)[-1]));
    return results;

a = np.random.normal(size = (4,3,10,40));
mask = np.random.randint(low = 0, high = 2, size = (4,1,10,40));
a = Dense2Sparse()([a, mask]);
b = np.random.normal(size = (4,3,40,20)).astype(np.float32);

print(SparseDenseMatMul()([a,b])); # eager mode is OK

inputs_a = tf.keras.Input((None, None, None), sparse = True);
inputs_b = tf.keras.Input((None, None, None));
results_c = SparseDenseMatMul()([inputs_a,inputs_b]);
model = tf.keras.Model(inputs = (inputs_a, inputs_b), outputs = results_c);
print(model([a,b])); # graph mode is failed