import random
from tensorflow import keras
from tensorflow.keras import layers

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    # Model has only one input so each data point has one element.
    yield [input_value]

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,100)
y = np.exp(-5/x)
e = np.random.randn(100)
y = y+0.05*e
plt.plot(x,y,'o')
plt.show()

x = x.reshape(-1,1)
y = y.reshape(-1,1)


def reg_model() : 
  activ = 'tanh'
    # Define the model architecture
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, input_shape=(x.shape), activation=activ),
    tf.keras.layers.Dense(100, activation=activ),
    tf.keras.layers.Dense(100, activation=activ),
    tf.keras.layers.Dense(100, activation=activ),
    tf.keras.layers.Dense(100, activation=activ),
    tf.keras.layers.Dense(100, activation=activ),
    tf.keras.layers.Dense(100, activation=activ),
    tf.keras.layers.Dense(1)
  ])

  # Train the digit classification model
  model.compile(optimizer='adam',
                loss=tf.keras.losses.mean_absolute_error)
  
  model.fit(
    x,
    y,
    epochs=20,
    validation_data=(x, y)
  ) 

  return model 

model = reg_model()

def representative_data_gen_reg():
# Approach 1 
  yield [x[40:80].reshape(40)] 
# Approach 2 
  # for i in x[40:80] : 
  #   yield {"val": i}
  
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen_reg

tflite_model_quant_wb = converter.convert()

def representative_data_gen_reg():
# Approach 1 
#yield [x[40:80].reshape(40)] 
# Approach 2 
  for i in x[70:80] : 
    yield {"dense_112_input": i.reshape(1,-1)}