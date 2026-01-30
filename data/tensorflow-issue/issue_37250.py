from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

number_of_cells = 2

model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(batch_input_shape=(1, 1, 1)))

cells = []

for _ in range(number_of_cells):
    cells.append(tf.keras.layers.GRUCell(10))

model.add(tf.keras.layers.RNN(cells, stateful=True))

model.compile()

model.save('rnn.tf', save_format='tf')

model2 = tf.keras.models.load_model('rnn.tf')