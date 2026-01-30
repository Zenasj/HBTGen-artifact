from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras

dataset: tf.data.Dataset = tf.data.experimental.make_csv_dataset('xor.csv', 4, label_name='result')
columns = [tf.feature_column.numeric_column('a'), tf.feature_column.numeric_column('b')]
input_column = keras.layers.DenseFeatures(columns)
layers = [input_column,
          keras.layers.Dense(4, activation='relu'),
          keras.layers.Dense(2)]
model = keras.Sequential(layers)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
model.fit(dataset, steps_per_epoch=4, epochs=2000)
keras.models.save_model(model, 'test_model.h5')
# model.save('test_model.h5')
model = keras.models.load_model('test_model.h5')