import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

class DataDepInit(tf.keras.layers.Layer):
  def __init__(self):
    super().__init__()

  def build(self, input_shape):
    # weight initialized by first batch
    self.w = self.add_weight(
        name="mean",
        shape=(1, 1, 1, 1),
        dtype=tf.float32,
        trainable=True,
        aggregation=tf.VariableAggregation.MEAN
    )
    # the controller about initialization
    self.initialized = self.add_weight(
        name="init",
        trainable=False,
        dtype=tf.bool,
    )

    self.initialized.assign(False)
    self.built = True
  
  def initialize(self, x):
    mean = tf.reduce_mean(x, axis=[0, 1, 2], keepdims=True)
    tf.print("initialize")
    self.w.assign(mean)

  def call(self, x):
    if not self.initialized:
      self.initialize(x)
      self.initialized.assign(True)
    return x - self.w

# ---------------------------------------------------------
with strategy.scope():
  x = tf.keras.Input(shape=(32, 32, 1))
  ddi = DataDepInit()
  model = tf.keras.Model(x, ddi(x))
  model.summary()
  
  def _step():
    model(tf.random.normal(shape=[128, 32, 32 , 1]))

  @tf.function
  def distributed_step():
    strategy.experimental_run_v2(_step, args=())

  for i in range(10):
    distributed_step()

import tensorflow as tf
# do not preallocate gpu memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import os

class MovingNorm(tf.keras.layers.Layer):
	def __init__(self, mean: float = 0.0, var: float = 1.0):
		super(MovingNorm, self).__init__()
		self.mean = mean
		self.var = var
	
	def build(self, input_shape):
		shape=(1, 1, 1, input_shape[3])
		self.moving_mean = self.add_weight('mean', shape, initializer=tf.keras.initializers.Constant(0.0), trainable=False, aggregation=tf.VariableAggregation.MEAN)
		self.moving_var = self.add_weight('var', shape, initializer=tf.keras.initializers.Constant(1.0), trainable=False, aggregation=tf.VariableAggregation.MEAN)
		self.target_mean = self.add_weight('target_mean', shape, initializer=tf.keras.initializers.Constant(self.mean))
		self.target_var = self.add_weight('target_var', shape, initializer=tf.keras.initializers.Constant(self.var))
	
	def call(self, x, decay: float):
		tf.print('decay: ', decay)
		if decay<1.0:
			mean, var = tf.nn.moments(x, [0, 1, 2], keepdims=True)
			var=tf.maximum(var, 1e-3)
			self.moving_mean.assign(decay*self.moving_mean + (1-decay)*mean)
			self.moving_var.assign(decay*self.moving_var + (1-decay)*(var**0.5))
		# output
		mult = self.target_var / self.moving_var
		x_norm = mult*x + (self.target_mean - self.moving_mean*mult)
		return x_norm

def train_batch(data):
	with tf.GradientTape() as tape:
		loss = tf.reduce_sum(net(data, 0.9)**2)
	x=net.trainable_variables
	gradients = tape.gradient(loss, x)
	optimizer.apply_gradients(zip(gradients, x))
	return loss

@tf.function
def dist_train(data):
	distributed_strategy.run(train_batch, (data,))

#= no NCCL on windows :(
if os.name=='nt':
	distributed_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
else:
	distributed_strategy = tf.distribute.MirroredStrategy()

with distributed_strategy.scope():
	net=MovingNorm()
	# need to run the network so that variables are created within distributed scope
	out = net(tf.random.normal((4,2,2,3)), 0.9)

with distributed_strategy.scope():
	optimizer = tf.keras.optimizers.Adam(1e-3, 0.9, 0.999, 1e-7, amsgrad=False)

ds = tf.data.Dataset.from_tensor_slices([tf.random.normal((4,2,2,3)) for _ in range(64)])
dist_ds = distributed_strategy.experimental_distribute_dataset(ds)

for data in dist_ds:
	dist_train(data)