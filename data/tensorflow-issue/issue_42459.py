from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=2, activation='softmax', name='output'))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=10),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
dummy_data_x = [[0, 0],
                [1, 0],
                [0, 1],
                [1, 1]]
dummy_data_y = [0, 1, 0, 1]
print(model.evaluate(x=dummy_data_x, y=dummy_data_y))
model.fit(x=dummy_data_x, y=dummy_data_y, epochs=10)
print(model.evaluate(x=dummy_data_x, y=dummy_data_y))
model.save('test_model')
model = tf.keras.models.load_model('test_model')
print(model.evaluate(x=dummy_data_x, y=dummy_data_y))