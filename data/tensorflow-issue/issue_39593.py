from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

#defining a simple subclass of keras model. Just one dense layer is present we will try to restore the kernel and bias of this layer using the `tf.train.Checkpoint`

class Net(tf.keras.Model):
  """A simple linear model."""

  def __init__(self):
    super(Net, self).__init__()
    self.l1 = tf.keras.layers.Dense(5)
  def call(self, x):
    return self.l1(x)

net = Net()

#initializing the net by running it on a demo input
_ = net(np.arange(4).reshape(-1,1)) #model has been initalized

#checking the weights
print(net.l1.bias,net.l1.kernel)

#saving the model parameters
ckpt = tf.train.Checkpoint(netin=net)
mgr = tf.train.CheckpointManager(ckpt,'./tf_ckpt',max_to_keep=1)
mgr.save()

#trying to load just the bias of the net's parameters
initializer = tf.keras.initializers.Constant(value=1.)
restore_bias_here = tf.Variable(initial_value=initializer(shape=(5,)))
ckpt_layer = tf.train.Checkpoint(bias = restore_bias_here)
ckpt_net = tf.train.Checkpoint(l1 = ckpt_layer)
ckpt1 = tf.train.Checkpoint(netin=ckpt_net)
#queing up the restores
ckpt1.restore(tf.train.latest_checkpoint('./tf_ckpt'))

#and now trying to load just the kernel but from 1. ckpt_net created previously
delayed_restore_kernel_here = tf.Variable(initial_value=initializer(shape=(1,5)))
print(delayed_restore_kernel_here) #all 1's tensor
ckpt_layer.kernel = delayed_restore_kernel_here
print(delayed_restore_kernel_here) #the variable is loaded up perfectly

#2. from #ckpt_layer created previously. But this wont succeed
delayed_restore_kernel_here = tf.Variable(initial_value=initializer(shape=(1,5)))
print(delayed_restore_kernel_here) #all 1's tensor
ckpt_net.l1.kernel = delayed_restore_kernel_here
print(delayed_restore_kernel_here) #all 1's tensor