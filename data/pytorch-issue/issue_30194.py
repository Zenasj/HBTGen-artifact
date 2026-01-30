import random

import numpy as np
X = np.random.randn(4, 2)
Y = np.random.randn(2)
Z = np.einsum("ab,b->ab", X, Y)

Y2 = Y.reshape(1, 2)
Z2 = np.einsum("ab,ab->ab", X, Y2)
assert np.all(Z == Z2)

import tensorflow as tf
X = tf.random.normal((4, 2))
Y = tf.random.normal((2,))
Z = tf.einsum("ab,b->ab", X, Y)

Y2 = tf.reshape(Y, (1, 2))
Z2 = tf.einsum("ab,ab->ab", X, Y2)
assert tf.reduce_all(Z == Z2)

import torch
X = torch.randn(4, 2)
Y = torch.randn(2)
Z = torch.einsum("ab,b->ab", X, Y)

Y2 = Y.reshape(1, 2)
Z2 = torch.einsum("ab,ab->ab", X, Y2)