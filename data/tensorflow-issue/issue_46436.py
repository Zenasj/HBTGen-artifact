import numpy as np
import random

#Scenario 1: Lower case accuracy (accuracy displayed is proper)

from tensorflow import keras
from tensorflow.keras import layers

X_train = np.random.random((100,8))
y_train = np.random.randint(0,2,(100,))

model = keras.Sequential(
    [
     layers.Dense(10,activation="relu",input_shape=(8,)),
     layers.Dense(10,activation="relu"),
     layers.Dense(1,activation="sigmoid")
    ]
)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=64,epochs=2,verbose=2)
print(model.metrics)

# Epoch 1/2
# 2/2 - 0s - loss: 0.6900 - accuracy: 0.4900
# Epoch 2/2
# 2/2 - 0s - loss: 0.6892 - accuracy: 0.5000   #<------
        
# [<tensorflow.python.keras.metrics.Mean object at 0x7f90e6626d90>, 
#  <tensorflow.python.keras.metrics.MeanMetricWrapper object at 0x7f90e5ac73a0>]  #<------

#Scenario 2: Uppercase case Accuracy (accuracy displayed / epoch always 0)

from tensorflow import keras
from tensorflow.keras import layers

X_train = np.random.random((100,8))
y_train = np.random.randint(0,2,(100,))

model = keras.Sequential(
    [
     layers.Dense(10,activation="relu",input_shape=(8,)),
     layers.Dense(10,activation="relu"),
     layers.Dense(1,activation="sigmoid")
    ]
)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['Accuracy'])
model.fit(X_train,y_train,batch_size=64,epochs=2,verbose=2)
print(model.metrics)

# Epoch 1/2
# 2/2 - 0s - loss: 0.7021 - accuracy: 0.0000e+00
# Epoch 2/2
# 2/2 - 0s - loss: 0.6999 - accuracy: 0.0000e+00   #<------
        
# [<tensorflow.python.keras.metrics.Mean object at 0x7f90c80e4bb0>, 
#  <tensorflow.python.keras.metrics.Accuracy object at 0x7f90ca039a30>]  #<------

#With lower case accuracy
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print(model.metrics)

#With upper case Accuracy
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['Accuracy'])
print(model.metrics)