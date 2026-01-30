import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class TestLayer2(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(TestLayer2, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dim)
        
    def call(self, x):
        x = self.dense1(x)
        return x

class TestLayer1(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(TestLayer1, self).__init__()
        self.list_of_dense = [TestLayer2(dim) for _ in range(2)]
        
    def call(self, x):
        for i in range(len(self.list_of_dense)):
            x = self.list_of_dense[i](x)
        return x
    
class TestModel(tf.keras.Model):
    def __init__(self, dim):
        super(TestModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dim)
        self.layer = TestLayer1(dim*2)
        
    def call(self, x):
        x = self.dense1(x)
        x = self.layer(x)
        return x

t_model = TestModel(512)

tmp = tf.random.normal([64,512])

t_model(tmp)

t_model.summary()

import tensorflow as tf

class TestLayer2(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(TestLayer2, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dim)
        
    def call(self, x):
        x = self.dense1(x)
        return x

class TestLayer1(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(TestLayer1, self).__init__()
        self.dense1 = TestLayer2(dim)
        self.dense2 = TestLayer2(dim)
        
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class TestModel(tf.keras.Model):
    def __init__(self, dim):
        super(TestModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dim)
        self.layer = TestLayer1(dim*2)
        
    def call(self, x):
        x = self.dense1(x)
        x = self.layer(x)
        return x

t_model = TestModel(512)

tmp = tf.random.normal([64,512])

t_model(tmp)

t_model.summary()

import tensorflow as tf

class TestLayer2(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(TestLayer2, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dim)
        
    def call(self, x):
        x = self.dense1(x)
        return x

class TestLayer1(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(TestLayer1, self).__init__()
        self.list_of_dense = [TestLayer2(dim) for _ in range(2)]
        
    def call(self, x):
        for i in range(len(self.list_of_dense)):
            x = self.list_of_dense[i](x)
        return x
    
class TestModel(tf.keras.Model):
    def __init__(self, dim):
        super(TestModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dim)
        self.layer = TestLayer1(dim*2)
        
    def call(self, x):
        x = self.dense1(x)
        x = self.layer(x)
        return x

t_model = TestModel(512)

tmp = tf.random.normal([64,512])

t_model(tmp)

len(t_model.trainable_variables)

gradients = tape.gradient(loss, t_model.trainable_variables)
optimizer.apply_gradients(list(zip(gradients, t_model.trainable_variables)))

class TestLayer1(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(TestLayer1, self).__init__()
       
        self.layer_names = ['layer_' + i for i in range(2)]
        for layer_name in self.layer_names:
             self.__setattr__(layer_name, TestLayer2(dim))
                
    def call(self, x):
        for layer_name in self.layer_names:
            x = self.__getattribute__(layer_name)(x)
        return x