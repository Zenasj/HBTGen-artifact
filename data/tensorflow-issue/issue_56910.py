from tensorflow.keras import layers

from tensorflow import keras
import numpy as np

loss_func = keras.losses.BinaryCrossentropy()
nn = keras.Sequential([
  keras.layers.Dense(2**8, input_shape=(1,), activation='relu'),
  keras.layers.Dense(2, activation='softmax')
])
nn.compile(loss=loss_func,optimizer='adam')
train_x = np.array([0.4])
train_y = np.array([[0,1]])
print(nn.predict(train_x))
print("Evaluated loss = ",nn.evaluate(train_x,train_y))
print("Function loss = ",loss_func(train_y,nn.predict(train_x)).numpy())
print("Manual loss = ",np.average( -train_y*np.log(nn.predict(train_x)) -(1-train_y)*np.log(1. - nn.predict(train_x)) ))

### Relevant log output