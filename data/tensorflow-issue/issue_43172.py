import numpy as np
import math
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

tf.random.set_seed(0)
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

loss_function = tf.keras.losses.BinaryCrossentropy()

l=[]
w=np.array([[-0.2373],[ 1.    ]], dtype=np.float16) #weights
b=np.array([0.], dtype=np.float16) #array of biases
l.append(w)
l.append(b)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=2, dtype=tf.float16))

opt = tf.keras.optimizers.SGD(learning_rate=0.1)
model.layers[0].set_weights(l)
print("Init weights")
print(model.get_weights())
model.compile(optimizer=opt, loss=loss_function) 
model.fit(x=x, y=y, epochs=1, batch_size=1, verbose=0)
print("Fitted weights")
print(model.get_weights())

loss_result = loss_function(y, model(x, training=False))

if math.isnan(loss_result):
    print('NAN Error')