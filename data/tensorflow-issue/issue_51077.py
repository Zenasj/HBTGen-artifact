import math
from tensorflow.keras import layers
from tensorflow.keras import models

class MyScore(Metric):
  def __init__(self, name='my_score', **kwargs):
    super().__init__(name=name, **kwargs)
    self.cost = self.add_weight(name='cost', initializer='zeros')
    self.misprediction_cost = self.add_weight(name='misprediction_cost', initializer='zeros')
    self.weighted_samples = self.add_weight(name='weighted_samples', initializer='zeros')
    self.score = self.add_weight(name='score', initializer='zeros')
    self.tp = self.add_weight(name='tp', initializer='zeros')
    self.tn = self.add_weight(name='tn', initializer='zeros')
    self.fp = self.add_weight(name='fp', initializer='zeros')
    self.fn = self.add_weight(name='fn', initializer='zeros')
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    matrix = tf.math.confusion_matrix(y_true, y_pred)
    self.tp += matrix[0][0]
    self.tn += matrix[0][1]
    self.fp += matrix[1][0]
    self.fn += matrix[1][1]
    self.cost.assign((self.tn + self.fp)/(self.tp + self.fn))
    self.misprediction_cost.assign(self.fn * self.cost + self.fp)
    self.weighted_samples.assign(self.tn + self.fp + self.cost * (self.tp + self.fn))
    self.score.assign(1.0 - self.misprediction_cost / self.weighted_samples)

  def result(self):
    return self.score

  def reset_state(self):
    self.tp.assign(0)
    self.tf.assign(0)
    self.fp.assign(0)
    self.fn.assign(0)
    self.cost.assign(0.0)
    self.misprediction_cost.assign(0.0)
    self.weighted_samples.assign(0.0)
    self.score.assign(0.0)

import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, InputLayer

def build_nn():
    model = Sequential([
      InputLayer(input_shape=(13,)),
      Dense(units=64, activation='relu'),
      BatchNormalization(),
      Dropout(0.25),
      Dense(units=32, activation='relu'),
      BatchNormalization(),
      Dense(units=1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[MyScore()], run_eagerly=True)
    return model

X = pd.read_csv('heart.csv')
y = X.pop('target')
model = build_nn()
history = model.fit(X, y, epochs=5)