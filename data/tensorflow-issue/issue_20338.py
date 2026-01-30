import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = []
        for i in range(20):
            self.layers.append(tf.keras.layers.Dense(units=10))
    def call(self, inputs):
        for i in range(20):
            x = self.layers[i](inputs)
        return x
input_layer = tf.keras.layers.Input(shape=(10,))
modules = MyModel()(input_layer)
model = tf.keras.Model(inputs=input_layer, outputs=modules)
model.summary()

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.iamlist = []
        for i in range(20):
            self.iamlist.append(tf.keras.layers.Dense(units=10))
    def call(self, inputs):
        for i in range(20):
            x = self.iamlist[i](inputs)
        return x
input_layer = tf.keras.layers.Input(shape=(10,))
modules = MyModel()(input_layer)
model = tf.keras.Model(inputs=input_layer, outputs=modules)
model.summary()