import random

conv = layers.Conv2D(1, 2, 1, autocast=False)
x = tf.random.normal([2, 1, 2, 2])
print(conv(x)) # no error

import tensorflow as tf
from keras import layers

class MyModule(tf.Module):
    def __init__(self):
        self.conv = layers.Conv2D(1, 2, 1, autocast=False)
    
    @tf.function
    def __call__(self, x):
        return self.conv(x)

if __name__ == '__main__':
    model = MyModule()

    tf.config.run_functions_eagerly(True)
    x = tf.random.normal([2, 1, 2, 2])
    print(model(x)) # tf.Tensor([], shape=(2, 0, 1, 1), dtype=float32)

    tf.config.run_functions_eagerly(False)
    x = tf.random.normal([2, 1, 2, 2])
    print(model(x)) # Error when tracing  
    model.__call__.get_concrete_function(x) # Same error if we call this instead of the last line

import tensorflow as tf

class MyModule(tf.Module):
    def __init__(self):
        self.kernel = tf.random.normal((2,2,2,1))
    
    @tf.function
    def __call__(self, x):
        return tf.raw_ops.Conv2D(input=x, filter=self.kernel, strides=[1,1,1,1], padding='VALID')

if __name__ == '__main__':
    model = MyModule()

    tf.config.run_functions_eagerly(True)
    x = tf.random.normal([2, 1, 2, 2])
    print(model(x)) # tf.Tensor([], shape=(2, 0, 1, 1), dtype=float32)

    tf.config.run_functions_eagerly(False)
    x = tf.random.normal([2, 1, 2, 2])
    print(model(x)) # ValueError when tracing  
    model.__call__.get_concrete_function(x) # Same error if we call this instead of the last line

import tensorflow as tf

class MyModule(tf.Module):
    def __init__(self):
        self.kernel = tf.random.normal((2,2,2,1))
    
    @tf.function
    def __call__(self, x):
        print(f'{tf.config.functions_run_eagerly() = }') # False
        return tf.raw_ops.Conv2D(input=x, filter=self.kernel, strides=[1,1,1,1], padding='VALID')

if __name__ == '__main__':
    model = MyModule()
    x = tf.random.normal([2, 1, 2, 2])
    print(model(x)) # ValueError