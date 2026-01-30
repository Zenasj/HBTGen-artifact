import random
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
from tensorflow import keras

# try-except eager trigger to prevent trouble from ipynb
try:
    tf.enable_eager_execution();
except:
    pass

def build_classifier_model():
    model = keras.Sequential([
        keras.layers.SimpleRNN(64, input_shape=(10, 10)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.tanh)
    ])

    optimizer = tf.optimizers.Adam(0.001) #tf.train.AdamOptimizer(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[keras.metrics.mae])
    return model

some_model = build_classifier_model();
  
def build_model():
    def custom_loss(y_true, y_pred):
      # simply tile y_pred as an example
      y_cred = tf.tile(tf.expand_dims(y_pred, axis=2), (1, 1, 10))
      print(y_cred.shape)
      return tf.reduce_sum(some_model(y_cred) ** 2)
    
    model = keras.Sequential([
        keras.layers.Dense(64, input_shape=(12,)),
        keras.layers.Dense(10, activation=tf.nn.tanh)
    ])

    optimizer = tf.optimizers.Adam(0.001) #tf.train.AdamOptimizer(0.001)

    model.compile(loss=custom_loss,
                optimizer=optimizer,
                metrics=[keras.metrics.mae])
    return model
        
m = build_model();
# fitting randomness as example
x = np.random.random((500, 12));
y = tf.cast(np.random.randint(0, 2, size=(500,)), tf.float32);
m.fit(x, y, epochs=10);