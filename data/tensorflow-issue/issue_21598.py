from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

Conv2D = tf.keras.layers.Conv2D
Conv2DTranspose = tf.keras.layers.Conv2DTranspose
Input = tf.keras.layers.Input
Model = tf.keras.Model

input = Input(shape=(None, None, 1), name='input')
x = Conv2DTranspose(8, (2, 2), dilation_rate=2, padding='valid')(input)
x = Conv2DTranspose(8, (3, 3), dilation_rate=2, padding='valid')(x)
x = Conv2DTranspose(8, (3, 3), dilation_rate=2, padding='valid')(x)

model_dilation = Model(inputs=input, outputs=x, name='dil')

x = Conv2DTranspose(8, (2, 2), dilation_rate=1, padding='valid')(input)
x = Conv2DTranspose(8, (3, 3), dilation_rate=1, padding='valid')(x)
x = Conv2DTranspose(8, (3, 3), dilation_rate=1, padding='valid')(x)

model_no_dilation = Model(inputs=input, outputs=x, name='no_dil')

output_dil = model_dilation.predict(np.zeros((1, 5, 5, 1)))
output_no_dil = model_no_dilation.predict(np.zeros((1, 5, 5, 1)))

print("shape dil", np.shape(output_dil))
print("shape no dil", np.shape(output_no_dil))