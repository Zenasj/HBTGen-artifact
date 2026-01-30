import tensorflow as tf
from tensorflow import keras

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()

  @tf.function(jit_compile=True) # Comment this line, it will succeed
  def call(self, x):
    pred = True
    x1 = x
    y = tf.cond(pred, lambda: x1, lambda: x1)
    return y

m = Model()
input_shape = [4]
x = tf.constant([4.,5.,6.,7.], shape=input_shape)
y = m(x)