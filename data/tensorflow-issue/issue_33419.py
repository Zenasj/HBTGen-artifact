import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import os
import tensorflow as tf
import tempfile
import glob

class A(tf.keras.models.Model):
    def __init__(self):
        self.something = tf.keras.backend.variable(name='something', value=1.0)
        super().__init__()

        
for fmt in ['tf', 'h5']:
    print(f'fmt={fmt}')
    filename_a = os.path.join(tempfile.mkdtemp(), 'data_{fmt}')
    filename_b = os.path.join(tempfile.mkdtemp(), 'data_{fmt}')

    a = A()
    a.save_weights(filename_a, save_format=fmt)
    a.something.assign(1.2)
    a.save_weights(filename_b, save_format=fmt)
    print('trainable_variables', a.trainable_variables)

    b = A()
    b.load_weights(filename_b)
    value = float(b.something.numpy())
    check = np.abs(value - 1.2) < 1e-4
    print(value, f'fmt={fmt}. class var PASS={check}. The value should be 1.2')
    
    value = float(b.trainable_variables[0].numpy())
    check = np.abs(value - 1.2) < 1e-4
    print(value, f'fmt={fmt}. trainable_variables PASS={check}. The value should be 1.2')

import os
import tensorflow as tf
import tempfile
import glob

class A(tf.keras.models.Model):
    def __init__(self):
        self.something = tf.Variable(1.0, dtype='float32', trainable=True)
        super().__init__()

        
for fmt in ['tf', 'h5']:
    print(f'fmt={fmt}')
    filename_a = os.path.join(tempfile.mkdtemp(), 'data_{fmt}')
    filename_b = os.path.join(tempfile.mkdtemp(), 'data_{fmt}')

    a = A()
    a.save_weights(filename_a, save_format=fmt)
    a.something.assign(1.2)
    a.save_weights(filename_b, save_format=fmt)
    print('trainable_variables', a.trainable_variables)

    b = A()
    b.load_weights(filename_b)
    value = float(b.something.numpy())
    check = np.abs(value - 1.2) < 1e-4
    print(value, f'fmt={fmt}. class var PASS={check}. The value should be 1.2')
    
    value = float(b.trainable_variables[0].numpy())
    check = np.abs(value - 1.2) < 1e-4
    print(value, f'fmt={fmt}. trainable_variables PASS={check}. The value should be 1.2')

import os
import tensorflow as tf
import tempfile
import glob

class A(tf.keras.models.Model):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.something = self.add_weight(initializer=tf.keras.initializers.Ones(), dtype=tf.float32, shape=(1,), name='something')

    def call(self, x):
        return x


for fmt in ['tf', 'h5']:
    print(f'fmt={fmt}')
    filename_a = os.path.join(tempfile.mkdtemp(), 'data_{fmt}')
    filename_b = os.path.join(tempfile.mkdtemp(), 'data_{fmt}')

    a = A()
    a(np.random.randn(3, 4).astype(np.float32))
    a.save_weights(filename_a, save_format=fmt)
    a.something.assign([1.2])
    a.save_weights(filename_b, save_format=fmt)
    print('trainable_variables', a.trainable_variables)

    b = A()
    b(np.random.randn(3, 4).astype(np.float32))
    b.load_weights(filename_b)
    value = float(b.something.numpy())
    check = np.abs(value - 1.2) < 1e-4
    print(value, f'fmt={fmt}. class var PASS={check}. The value should be 1.2')
    
    value = float(b.trainable_variables[0].numpy())
    check = np.abs(value - 1.2) < 1e-4
    print(value, f'fmt={fmt}. trainable_variables PASS={check}. The value should be 1.2')

import numpy as np
import os
import tensorflow as tf
import tempfile
import glob

class ScalarLayer(tf.keras.layers.Layer):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = tf.Variable(value, dtype='float32', trainable=True)

class A(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.hack = ScalarLayer(1.0)
        self.something = self.hack.value

for fmt in ['tf', 'h5']:
    print(f'fmt={fmt}')
    filename_a = os.path.join(tempfile.mkdtemp(), 'data_{fmt}')
    filename_b = os.path.join(tempfile.mkdtemp(), 'data_{fmt}')

    a = A()
    a.save_weights(filename_a, save_format=fmt)
    a.something.assign(1.2)
    a.save_weights(filename_b, save_format=fmt)
    print('trainable_variables', a.trainable_variables)

    b = A()
    b.load_weights(filename_b)
    value = float(b.something.numpy())
    check = np.abs(value - 1.2) < 1e-4
    print(value, f'fmt={fmt}. class var PASS={check}. The value should be 1.2')

    value = float(b.trainable_variables[0].numpy())
    check = np.abs(value - 1.2) < 1e-4
    print(value, f'fmt={fmt}. trainable_variables PASS={check}. The value should be 1.2')