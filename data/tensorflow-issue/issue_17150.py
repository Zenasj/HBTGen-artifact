import random
import tensorflow as tf
from tensorflow.keras import layers

import keras  # also works for tf.keras
import numpy as np

model = keras.Sequential()
model.add(keras.layers.Dense(10, input_shape=(5, 6)))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

x_data = np.random.random((32, 5, 6))
y_data = np.random.randint(0, 9, size=(32,))

model.fit(x=x_data, y=y_data, batch_size=16, epochs=3)

from tensorflow import keras
import numpy as np

dense_layer = keras.layers.Dense(10, activation='softmax')

model = keras.Sequential()
model.add(keras.layers.Embedding(11, 6, mask_zero=True, embeddings_initializer=keras.initializers.Constant(value=0.1)))
model.add(dense_layer)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

# Set the weights for our dense layer such that each category has the same
# probability except for 1, which will have a lower probability
new_weights = []
for weight in dense_layer.get_weights():
    if weight.shape == (6,10):
        weight = np.full_like(weight, 0.1, dtype='float32')
        weight[:,1] = -2.
    new_weights.append(weight)
dense_layer.set_weights(new_weights)

# Create random dataset where each sample has a different length with 0 padding
# and none of the labels use category 1
x_data = np.zeros((32, 20))
y_data = np.zeros((32, 20, 1))
for i in range(32):
    length = np.random.randint(1, 20)
    x_data[i, 0:length] = np.random.randint(2, 10, size=length)
    y_data[i, 0:length] = np.random.randint(2, 10, size=(length, 1))

# Evaluate to get the cost
cost1 = model.evaluate(x_data, y=y_data)

# Change the 0s to 1s in the labels. If masking is handled correctly, this
# shouldn't matter as these labels will be ignored anyway. If masking is not
# handled correctly, then the cost should go up as the probability of 1 being
# the correct category is lower than any other category
y_data[y_data == 0] = 1

cost2 = model.evaluate(x_data, y=y_data)

assert cost1 == cost2