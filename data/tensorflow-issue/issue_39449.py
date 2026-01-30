import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

def generate_sparse_matrix():
    return tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1.0, 2.0], dense_shape=[3, 4])

input_model = layers.Input(shape=(8,))
dense_layer = layers.Dense(3, activation='relu')(input_model)
sparse_dense_matmul = lambda x: tf.sparse.sparse_dense_matmul(x[0], x[1], adjoint_a=True, adjoint_b=True)
multiplied = layers.Lambda(sparse_dense_matmul)((generate_sparse_matrix(), dense_layer))
multiplied = tf.transpose(multiplied)

out_layer = layers.Dense(1, activation='sigmoid', dtype='float32')(multiplied)

model = models.Model(input_model, out_layer)