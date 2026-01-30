import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Net(tf.keras.Model):
  """A simple linear model."""

  def __init__(self):
    super(Net, self).__init__()
    #self.l1 = tf.keras.layers.Dense(5)
  def build(self,input_shape):
    self.l1 = tf.keras.layers.Dense(5)
    self.dummy = tf.Variable(trainable=True,initial_value=tf.keras.initializers.glorot_normal()(shape=(1,),dtype=tf.float32))
    print('built layers')
  def call(self, x):
    return self.l1(x)

net = Net()
net.build([1,])
net.save_weights('easy_checkpoint')