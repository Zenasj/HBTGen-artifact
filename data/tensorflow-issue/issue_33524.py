from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import tensorflow.keras.losses as losses
from tensorflow.keras.optimizers import Adam
import numpy as np

def CreateModel():
  model = Sequential()
  model.add(Conv2D(filters=32, kernel_size=3, name='images', activation='relu', kernel_initializer='glorot_normal', input_shape=(39, 39, 3)))
  model.add(Flatten())
  model.add(Dense(10, bias_initializer='zeros', kernel_initializer='glorot_normal'))

  return model

def generatorFunction():
  while 1:
    s = (39, 39, 3)
    z = np.zeros(s, dtype=np.float64)

    zo = np.zeros(10, dtype=np.float64)
    yield z, zo

def RunScript():
  # Setup session options
  config = tf.ConfigProto()

  if tf.test.is_gpu_available():
    config.gpu_options.allow_growth = True

  session = tf.Session(config=config)
  session.run(tf.global_variables_initializer())
  K.set_session(session)

  # Setup strategy
  cross_device_ops = tf.distribute.ReductionToOneDevice(reduce_to_device='/device:CPU:0')
  mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)

  # Setup data
  trainingDataPrepped = tf.data.Dataset.from_generator(
    generatorFunction,
    output_types=(tf.float32, tf.float32),
    output_shapes=((39,39,3), (10))).batch(32)

  vDataPrepped = tf.data.Dataset.from_generator(
    generatorFunction,
    output_types=(tf.float32, tf.float32),
    output_shapes=((39,39,3), (10))).batch(32)

  # Create and train model
  with mirrored_strategy.scope():
    model = CreateModel()
    lossType = losses.mean_squared_error
    model.compile(optimizer=Adam(), loss=lossType)


    model.fit(trainingDataPrepped,
              validation_data=vDataPrepped,
              validation_steps=1,
              epochs=3000,
              shuffle=True,
              steps_per_epoch=1)

if __name__ == "__main__":
  RunScript()