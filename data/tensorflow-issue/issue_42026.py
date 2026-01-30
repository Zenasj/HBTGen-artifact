from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self,**kwargs):
    layer1 = tf.keras.layers.Dense(10)
    # define input shape, and reinit ...
    inputs = tf.keras.Input(shape=(5,))
    super(MyModel,self).__init__(inputs=inputs, outputs=layer1(inputs),**kwargs)
    # calling 'summary' will show specific input shapes
    self.summary()


m = MyModel()
m = MyModel()