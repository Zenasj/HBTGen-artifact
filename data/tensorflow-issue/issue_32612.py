from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

print(tf.__version__)
class MyCustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        return 1.



a = tf.keras.layers.Input(shape=(32,))
b = tf.keras.layers.Dense(32)(a)
model = tf.keras.models.Model(inputs=a, outputs=b)
model.save('./model.h5') # save first and then compile
model.compile('sgd', MyCustomLoss())

# load the model
model_new = tf.keras.models.load_model('./model.h5')
model_new.compile('sgd', MyCustomLoss())

import tensorflow as tf

print(tf.__version__)
class MyCustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        return 1.



a = tf.keras.layers.Input(shape=(32,))
b = tf.keras.layers.Dense(32)(a)
model = tf.keras.models.Model(inputs=a, outputs=b)
model.compile('sgd', MyCustomLoss()) # first compile, then fit, then save is the normal order.
model.fit(some_data, epochs=1) 
model.save('./model.h5') # I can not save any earlier!!!
# load the model
model_new = tf.keras.models.load_model('./model.h5')
model_new.compile('sgd', MyCustomLoss())

# Custom Loss1 (for example) 
@tf.function() 
def customLoss1(yTrue,yPred):
  return tf.reduce_mean(yTrue-yPred)