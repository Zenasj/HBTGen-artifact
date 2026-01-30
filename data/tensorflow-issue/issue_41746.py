import random

import tensorflow as tf
import numpy as np

class Sampler:
    def __init__(self, sample_size=10):
        self.sample_size = tf.Variable(sample_size, dtype=tf.int32)
        self.samples = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
     
    @tf.function
    def get_new_samples(self, data):
        size = tf.shape(data)[0]
        new_samples = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        
        for i in range(size):
            if self.samples.size() < self.sample_size:
                self.samples.write(i, data[i,:])
                new_samples.write(i, data[i, :])
            else:
                if (tf.random.uniform([1]) > 0.5):
                    idx = np.random.randint(0, size)
                    new_sample = self.samples.read(idx)
                    self.samples.write(idx, data[i, :])
                    new_samples.write(i, new_sample)
                else:
                    new_samples.write(i, data[i, :])
        return new_samples.stack()
        
    
    def __call__(self, data):
        return tf.cond(tf.equal(self.sample_size, 0),
                      true_fn=lambda: data,
                      false_fn=self.get_new_samples(data))

s = Sampler()
s(tf.convert_to_tensor(np.random.rand(5, 3).astype(np.float32)))

import tensorflow as tf
import numpy as np

pool1 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
pool2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
new_items = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

def write_to_array(array, idx, item):
    array = array.write(idx, item)
    return array

@tf.function
def sampler(items, array1, array2):
    num_items = tf.shape(items)[0]
    for i in tf.range(num_items):
        if array1.size() < 10:
            array1 = write_to_array(array1, i, items[i, :, :, :])
            array2 = write_to_array(array2, i, items[i, :, :, :])
        else:
            array2 = write_to_pool(array2, i, items[i, :, :, :])
    return array2.stack()

items = tf.convert_to_tensor(np.random.rand(2, 2, 2, 3).astype(np.float32))
sampler(items, pool1, new_items)

import tensorflow as tf
import numpy as np

pool1 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
pool2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

def write_to_array(array, idx, item):
    array = array.write(idx, item)
    return array

@tf.function
def sampler(items, array1):
    num_items = tf.shape(items)[0]
    new_items = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    for i in tf.range(num_items):
        if array1.size() < 10:
            array1 = write_to_array(array1, i, items[i, :, :, :])
            new_items = write_to_array(new_items, i, items[i, :, :, :])
        else:
            new_items = write_to_array(new_items, i, items[i, :, :, :])
    return new_items.stack()

items = tf.convert_to_tensor(np.random.rand(2, 2, 2, 3).astype(np.float32))
sampler(items, pool1, new_items)