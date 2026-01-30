from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

# saving succeeds for number_of_cells = 1, but fails for number_of_cells > 1
number_of_cells = 2

model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(shape=(1, 1,)))

cells = []

for i in range(number_of_cells):
    cells.append(tf.keras.layers.GRUCell(10))

model.add(tf.keras.layers.RNN(cells))

model.save("rnn.h5")