import random
from tensorflow import keras
from tensorflow.keras import layers

import sys
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

#GPU growth code
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

#---------------PARAMS------------
batch_size = 32
max_out_len = 200
num_hidden = 400
num_classes = 73
max_time_steps = 900
num_features = 240
#---------------------------------

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.forward_cell = [
            tf.keras.layers.LSTMCell(num_hidden, kernel_initializer='glorot_uniform')]
        self.state = tf.keras.layers.RNN(self.forward_cell, time_major=False, return_sequences=True)
        self.dense = tf.keras.layers.Dense(num_classes, use_bias=True, kernel_initializer=tf.initializers.TruncatedNormal(mean=0, stddev=0.1))

    def call(self, inputs):
        x, seq_len = inputs
        mask = tf.sequence_mask(seq_len, maxlen=max_time_steps)
        state = self.state(inputs=x, mask=mask)
        logits = self.dense(state)
        return logits

x = tf.random.normal(shape=(2, batch_size, max_time_steps, num_features))
seq_len = tf.constant([[5]*batch_size,[7]*batch_size])
y_i = [tf.constant(sum([[[i,0],[i,1],[i,2]] for i in range(batch_size)], [])),tf.constant(sum([[[i,0],[i,2],[i,4]] for i in range(batch_size)], []))]
y_v = [tf.constant([1,2,4]*batch_size),tf.constant([5,1,2]*batch_size)]
v = [tf.constant([1,0]*(batch_size//2)),tf.constant([0,1]*(batch_size//2))]
def fn():
    for i in range(2):
        yield x[i], seq_len[i], y_i[i], y_v[i], v[i]

def loss(logits, s, yi, yv):
    y = tf.SparseTensor(yi, yv, tf.TensorShape([batch_size, max_out_len]))
    return tf.cast(tf.sparse.reduce_sum(y, axis=-1), tf.float32) - tf.reduce_sum(logits, axis=[-1, -2])


with tf.device('/cpu:0'):
    model = Model()
    model.build([(batch_size, max_time_steps, num_features), (batch_size,)])
    dataset = tf.data.Dataset.from_generator(fn, output_types=(tf.float32, tf.int32, tf.int64, tf.int32, tf.float32))

def run_eager(x, s, yi, yv, v):
    with tf.device('/gpu:0'):
        with tf.GradientTape() as tape:
            logits = model([x, s])
            losses = tf.reduce_sum(loss(logits, s, yi, yv))
        grads = tape.gradient(losses, model.trainable_variables)

    # ToDo: call optimizer.apply_gradients

    return losses

if len(sys.argv) > 1 and sys.argv[1] == 'func':
    run = tf.function(run_eager)
else:
    run = run_eager

with tf.device('/cpu:0'):
    it = iter(dataset)
    l = run(*next(it))

print(l)

import sys
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

#GPU growth code
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

#---------------PARAMS------------
batch_size = 32
max_out_len = 200
num_hidden = 400
num_classes = 73
max_time_steps = 900
num_features = 240
#---------------------------------

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        # self.forward_cell = [
        #     tf.keras.layers.LSTMCell(num_hidden, kernel_initializer='glorot_uniform')]
        # self.state = tf.keras.layers.RNN(self.forward_cell, time_major=False, return_sequences=True)
        self.state = tf.keras.layers.LSTM(num_hidden, kernel_initializer='glorot_uniform', time_major=False, return_sequences=True)
        self.dense = tf.keras.layers.Dense(num_classes, use_bias=True, kernel_initializer=tf.initializers.TruncatedNormal(mean=0, stddev=0.1))

    def call(self, inputs):
        x, seq_len = inputs
        mask = tf.sequence_mask(seq_len, maxlen=max_time_steps)
        state = self.state(inputs=x, mask=mask)
        logits = self.dense(state)
        return logits

x = tf.random.normal(shape=(2, batch_size, max_time_steps, num_features))
seq_len = tf.constant([[5]*batch_size,[7]*batch_size])
y_i = [tf.constant(sum([[[i,0],[i,1],[i,2]] for i in range(batch_size)], [])),tf.constant(sum([[[i,0],[i,2],[i,4]] for i in range(batch_size)], []))]
y_v = [tf.constant([1,2,4]*batch_size),tf.constant([5,1,2]*batch_size)]
v = [tf.constant([1,0]*(batch_size//2)),tf.constant([0,1]*(batch_size//2))]
def fn():
    for i in range(2):
        yield x[i], seq_len[i], y_i[i], y_v[i], v[i]

def loss(logits, s, yi, yv):
    y = tf.SparseTensor(yi, yv, tf.TensorShape([batch_size, max_out_len]))
    return tf.cast(tf.sparse.reduce_sum(y, axis=-1), tf.float32) - tf.reduce_sum(logits, axis=[-1, -2])


with tf.device('/cpu:0'):
    model = Model()
    model.build([(batch_size, max_time_steps, num_features), (batch_size,)])
    dataset = tf.data.Dataset.from_generator(fn, output_types=(tf.float32, tf.int32, tf.int64, tf.int32, tf.float32))

@tf.function
def run_eager(x, s, yi, yv, v):
    with tf.device('/gpu:0'):
        with tf.GradientTape() as tape:
            logits = model([x, s])
            losses = tf.reduce_sum(loss(logits, s, yi, yv))
        grads = tape.gradient(losses, model.trainable_variables)

    # ToDo: call optimizer.apply_gradients

    return losses

if len(sys.argv) > 1 and sys.argv[1] == 'func':
    run = tf.function(run_eager)
else:
    run = run_eager

with tf.device('/cpu:0'):
    it = iter(dataset)
    l = run(*next(it))

print(l)