import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
model = keras.Sequential([
keras.layers.ReLU(threshold=3,  input_shape=(2,))])
x = tf.constant([[[[1,2,3,4,5]]]])
print (np.array2string(model(x).numpy(), separator=', '))
#print (np.array2string(model.predict(x), separator=', '))