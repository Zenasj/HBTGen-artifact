import random

pool = multiprocessing.Pool()
dataset = tf.contrib.data.Dataset.from_generator(
    lambda: pool.imap(some_function, some_data), dtypes, shapes
)

pool = multiprocessing.Pool()
dataset = tf.contrib.data.Dataset.from_generator(
    pool.imap(some_function, some_data), dtypes, shapes
)

def generator():
    while True:
        yield np.zeros(2, np.float32)
        
dataset = tf.contrib.data.Dataset.from_generator(generator, (tf.float32, tf.float32))
x, y = dataset.make_one_shot_iterator().get_next()
session = tf.Session()
session.run([x, y])

def generator():
    while True:
        yield tuple(np.zeros(2, np.float32))
        
dataset = tf.contrib.data.Dataset.from_generator(generator, (tf.float32, tf.float32))
x, y = dataset.make_one_shot_iterator().get_next()
session = tf.Session()
session.run([x, y])

import tensorflow as tf
from tensorflow.contrib import data as tfdata
import numpy as np
from time import time

num_batches = 1000
batch_size = 100

class Generator:
    def __init__(self):
        self.times = []
    
    def __iter__(self):
        while True:
            x = np.random.normal()
            y = 3 + 5 * x
            x, y = np.asarray([x, y], np.float32)
            self.times.append(time())
            yield x, y

generator_state1 = Generator()

dataset = tfdata.Dataset.from_generator(
    lambda: generator_state1, 
    (tf.float32, tf.float32),
    (tf.TensorShape([]), tf.TensorShape([]))
)
prefetched = dataset.prefetch(3 * batch_size)
batches = prefetched.batch(batch_size)
iterator = batches.make_one_shot_iterator()

x, y = iterator.get_next()

w = tf.Variable([0, 0], dtype=tf.float32)
prediction = w[0] + w[1] * x
loss = tf.losses.mean_squared_error(y, prediction)
optimizer = tf.train.AdamOptimizer(0.1)
train_op = optimizer.minimize(loss)
init_op = tf.global_variables_initializer()

session = tf.Session()
session.run(init_op)

losses = []

start = time()
for _ in range(num_batches):
    _, _loss = session.run([train_op, loss])
    losses.append(_loss)
time() - start  # about seven seconds

losses = []

generator_state2 = Generator()
iterator = iter(generator_state2)

start = time()
for _ in range(num_batches):
    _x, _y = np.transpose([next(iterator) for _ in range(batch_size)])
    _, _loss = session.run([train_op, loss], {x: _x, y: _y})
    losses.append(_loss)
time() - start  # about one second

np.mean(np.diff(generator_state1.times))  # 7.1533812683949508e-05
np.mean(np.diff(generator_state2.times))  # 1.0633696558370612e-05

import tensorflow as tf

range5 = tf.contrib.data.Dataset.range(5).make_one_shot_iterator()
with tf.Session() as sess:
    print(tf.__version__)
    def _dataset_generator():
        while True:
            try:
                yield sess.run(range5.get_next())
            except tf.errors.OutOfRangeError:
                return
    gen_dataset = tf.contrib.data.Dataset.from_generator(_dataset_generator,
                                                         tf.int64)
    gen_it = gen_dataset.make_one_shot_iterator()
    while True:
        try:
            print(sess.run(gen_it.get_next()))
        except tf.errors.OutOfRangeError:
            break

var_len_dataset = ...  # Element shape (?, x, y, z)
all_slices = var_len_dataset.flat_map(
    lambda x: tf.data.Dataset.from_tensor_slices(x))  # Element shape (x, y, z)

batches = dataset.batch(BATCH_SIZE).prefetch(PREFETCH_FACTOR * BATCH_SIZE)

batches = dataset.apply(map_and_batch(lambda a, b: (a, b), BATCH_SIZE)).prefetch(PREFETCH_FACTOR * BATCH_SIZE)

import time

class PyDataSet:
    def __len__(self):
        return 20
        
    def __getitem__(self, item):
        time.sleep(0.5)  # <----------------- Takes 0.5 seconds to "load" the data
        return np.random.normal(size=(10))

ds = from_indexable(PyDataSet(), output_types=tf.float64, output_shapes=[None, 10], num_parallel_calls=4)
ds = ds.batch(9)
it = ds.make_one_shot_iterator()
entry = it.get_next()
with tf.Session() as sess:
    start = time.perf_counter()
    try:
        while True:
            print(sess.run(entry).shape)
    except tf.errors.OutOfRangeError:
        pass
    delta = time.perf_counter() - start
    print(delta)  # 2.5277008840057533