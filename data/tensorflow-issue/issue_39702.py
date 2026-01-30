from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras import Model

x_data = np.zeros((32, 28, 28))
x_mask = np.zeros((32, 10))
y = np.zeros((32, 10))

input_data = Input(shape=(28, 28))
output= Flatten()(input_data)
output = Dense(64, activation='relu')(output)
output = Dense(10)(output)
model = Model(inputs=input_data, outputs=output)


class MyLoss(tf.keras.losses.Loss):
  def call(self, y_true, y_pred):
      return (y_true - y_pred) ** 2  


model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=MyLoss(name='loss'),
    metrics=['accuracy'])

for i in range (2):
    print(i)
    loss = model.train_on_batch(x_data, y, sample_weight=x_mask)