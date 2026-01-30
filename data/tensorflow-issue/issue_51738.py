from tensorflow import keras 
from keras import layers
model = keras.Sequential()
model.add(layers.Dense(1,input_shape=(10, 1)))
model.summary()