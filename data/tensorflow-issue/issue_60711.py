import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf

class GradientDescent(tf.keras.optimizers.experimental.Optimizer):
  def __init__(self, learning_rate = 0.01, name='GDST'):
    super().__init__(name=name)
    #self.learning_rate = learning_rate
    self._learning_rate = self._build_learning_rate(learning_rate)
    self.temp = None
  
  def build(self, var_list):
    super().build(var_list)
  
  def update_step(self, gradient, variable):
    lr = tf.cast(self._learning_rate, gradient.dtype)
    output = tf.clip_by_value(self._learning_rate*gradient, clip_value_max = gradient.dtype.max, clip_value_min = gradient.dtype.min)
    variable.assign_sub(output)
    self.temp = output


  def get_config(self):
    return super().get_config()

opt = GradientDescent(learning_rate = 0.0001)

input_shape = 10000
output_shape = 100

def return_model(input_shape = 10000, output_shape = 500):
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(input_shape, activation='relu'),
    tf.keras.layers.Dense(10000, activation='relu'),
    #tf.keras.layers.Dense(10000, activation='tanh'),
    #tf.keras.layers.Dense(10000, activation='linear'),
    #tf.keras.layers.Dense(10000, activation='tanh'),
    tf.keras.layers.Dense(output_shape, activation='linear')
])
  return model


# Compile the model
#model.compile(optimizer=opt, loss='mse')

# Create a random dataset
x_train = tf.random.uniform(shape=[10000, 10000])
y_train = tf.random.uniform(shape=[10000, 500])

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
  model = return_model()
  model.compile(optimizer=opt, loss='mse')

  model(x_train[:2])
  model.summary()
  model.fit(x_train, y_train, epochs = 2, batch_size = 1024)
  opt.temp