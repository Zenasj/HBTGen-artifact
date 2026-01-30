from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.layers import (Dense, Input, Lambda)
from tensorflow.keras.models import Model, Sequential
from scipy import sparse
import numpy as np


def layer_lambda(input_x):
    sparse = input_x[0]
    dense = input_x[1]
    dense = tf.transpose(dense)
    y = tf.sparse.sparse_dense_matmul(sparse, dense)
    return tf.transpose(y)


dense_mat = np.eye(30, 30, dtype=np.float32)
sparse_mat = sparse.coo_matrix(dense_mat)
sparse_indices = np.mat([sparse_mat.row, sparse_mat.col]).transpose()
sparse_tensor = tf.SparseTensor(sparse_indices, sparse_mat.data, sparse_mat.shape)

model = Sequential()
model_input = Input(shape=(20,))
x = Dense(20)(model_input)
x = Dense(30)(x)
x = Lambda(layer_lambda, output_shape=(None, 30, 30))([sparse_tensor, x])
model = Model(model_input, x)

model.predict([[np.ones(20)]])

model.save("model.h5")

print("Save successfull")
print("loading ...")
model_load = tf.keras.models.load_model("model.h5", custom_objects={'layer_lambda': layer_lambda})

print("Load successfull")

def layer_lambda(input_x):
  dense_mat = np.eye(30, 30, dtype=np.float32)
  sparse_mat = sparse.coo_matrix(dense_mat)
  sparse_indices = np.mat([sparse_mat.row, sparse_mat.col]).transpose()
  sparse_tensor = tf.SparseTensor(sparse_indices, sparse_mat.data, sparse_mat.shape)
  dense = input_x
  dense = tf.transpose(dense)
  y = tf.sparse.sparse_dense_matmul(sparse_tensor, dense)
  return tf.transpose(y)

model = Sequential()
model_input = Input(shape=(20,))
x = Dense(20)(model_input)
x = Dense(30)(x)
x = Lambda(layer_lambda, output_shape=(None, 30, 30))(x)
model = Model(model_input, x)