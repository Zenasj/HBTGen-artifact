from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def build_model():
  input = tf.keras.layers.Input(shape=(2,))
  pred = tf.keras.layers.Dense(2, activation='softmax')(input)
  model = tf.keras.models.Model(inputs=input, outputs=pred)
  model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
  return model

X = np.array([[1,2],[3,1]])
Y = np.array([[1,0], [0,1]])
model = build_model()
model.fit(X, Y)
print(model.predict(X))  # this works

model_wrapped = KerasClassifier(build_model)
model_wrapped.fit(X, Y)
model_wrapped.predict(X)  # this crashes