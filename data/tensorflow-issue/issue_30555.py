from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "uint8")
target_data = np.array([[0],[1],[1],[0]], "uint8")

model = Sequential()
model.add(Dense(16, input_dim=2, use_bias=False, activation='relu'))
model.add(Dense(1, use_bias=False, activation='sigmoid'))

session = tf.keras.backend.get_session()
tf.contrib.quantize.create_training_graph(session.graph)
session.run(tf.global_variables_initializer())

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(training_data, target_data, nb_epoch=1000, verbose=2)
print(model.predict(training_data).round())
model.summary()

saver = tf.train.Saver()
saver.save(keras.backend.get_session(), 'xor-keras.ckpt')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(16, input_dim=2, use_bias=False, activation='relu'))
model.add(Dense(1, use_bias=False, activation='sigmoid')) 


 #<Load the model into a new session>

session = tf.keras.backend.get_session()

saver = tf.train.Saver()
saver.restore(session, 'xor-keras.ckpt')

tf.contrib.quantize.create_eval_graph(session.graph)

tf.io.write_graph(session.graph, '.', 'xor-keras-eval.pb', as_text=False)