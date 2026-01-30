from tensorflow import keras
from tensorflow.keras import optimizers

import math
import numpy as np
import tensorflow as tf

error = True
n_features = 100
batch = 2

"""
model
"""
x = tf.keras.Input(shape=(n_features,), dtype=tf.float32)
w = tf.Variable([1.0] * n_features)
b = tf.Variable(1.0)
z = tf.reduce_sum(w * x, axis=1, keepdims=True) + b

"""
loss is incorrect if error is true
"""
if error:
    y_ = tf.sigmoid(z)
else:
    y_ = 1.0 / (1.0 + math.e ** (-z))

m = tf.keras.Model(inputs=x, outputs=y_)

"""
loss
"""
optimizer=tf.keras.optimizers.SGD(learning_rate=0.001)
loss = tf.keras.losses.BinaryCrossentropy()
m.compile(optimizer = optimizer, loss = loss)

"""
train dataset
"""
x = np.array([[1.0 for i in range(n_features)]] * batch, dtype=np.float32)
y = np.array([0.0] * batch, dtype=np.float32)

"""
get correct loss
"""
logits = m(x)
l = loss(y, logits)

"""
get incorrect loss
"""
history = m.fit(x, y)

"""
history.history['loss'] != l.numpy()
"""
print(history.history)
print(l.numpy())