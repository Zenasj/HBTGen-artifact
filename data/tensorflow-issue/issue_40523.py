import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf

model = Sequential([
    Embedding(input_dim = 11,
              output_dim = 100,
              mask_zero = True,
              batch_input_shape = [None, None]),
    LSTM(1024, stateful = False,
         return_sequences = True),
    TimeDistributed(Dense(11, activation = 'softmax'))])
X, Y = [], []
for _ in range(10000):
    v1 = tf.random.uniform((50,), minval = 1, maxval = 11,
                           dtype = tf.int32)
    v2 = tf.zeros((10,), dtype = tf.int32)
    sample = tf.concat((v1, v2), axis = 0)
    X.append(sample[:-1])
    Y.append(sample[1:])
optimizer = RMSprop(learning_rate = 0.01)
model.compile(
    optimizer = optimizer,
    loss = 'sparse_categorical_crossentropy',
    metrics = ['sparse_categorical_accuracy'])
print('fitting')
model.fit(X, Y, epochs = 30)