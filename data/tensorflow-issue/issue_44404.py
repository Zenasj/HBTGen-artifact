import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

#x_train, y_train, x_test, y_test -- All defined before with images size 28 28 
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
predictions = model(x_train[:1]).numpy()
predictions

x = img.crop([matrix[0][0]-2, matrix[0][1], matrix[2][0]+2, matrix[2][1]+2])
img = x.resize((28, 28))
display(img.convert('LA'))
lstOfArrays.append(numpy.array(img.convert('LA')))