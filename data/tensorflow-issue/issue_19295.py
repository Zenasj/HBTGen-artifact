import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def model_fn(features, labels, mode):
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(64),
      tf.keras.layers.Dense(40, activation=tf.nn.softmax)
  ])
  
  logits = model(features)
  return tf.contrib.estimator.multi_class_head(...).create_estimator_spec(
      mode=mode, features=features, labels=labels, logits=logits, optimizer=tf.train.AdamOptimizer())