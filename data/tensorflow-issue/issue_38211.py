from tensorflow import keras
from tensorflow.keras import layers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

class FooLayer(tf.keras.layers.Layer):
    def __init__(self, siz):
        super(FooLayer, self).__init__()
        self.siz = siz
        self.buildFoo(siz)

    def call(self, in_data):
        Foo0 = tf.multiply(in_data,self.FooTns0)
        FooList = []
        FooList.append(Foo0)
        for it in range(1,self.siz+1):
            tmp = tf.multiply(FooList[it-1],self.FooTns[it-1])
            FooList.append(tmp)
        return FooList[self.siz]

    def buildFoo(self,siz):
        self.FooTns0 = tf.Variable(1, name="TNS0")
        self.FooTns = []
        for it in range(0,self.siz):
            self.FooTns.append(tf.Variable(it, name="TNS"+str(it+1)))

class FooModel(tf.keras.Model):
    def __init__(self, siz):
        super(FooModel, self).__init__()
        self.flayer = FooLayer(siz)

    def call(self, in_data):
        return self.flayer(in_data)

model = FooModel(5)

for v in model.trainable_variables:
    print(v.name)

for v in model.variables:
    print(v.name)

TNS0:0
TNS0:0