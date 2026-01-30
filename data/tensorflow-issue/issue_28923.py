import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

feature_columns = []
headers = dataframe.columns.tolist()

# feature cols
for header in headers:
    temp = feature_column.numeric_column(header)
    feature_columns.append(feature_column.bucketized_column(temp, boundaries=[-89, -70, -65, -60, -55, -50, -40, -30, -20]))
    #feature_columns.append(temp) #old code used without buckets. only numeric columns


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(3, activation='sigmoid') #cambiare con il numero di zone
])

for header in headers:
    temp = feature_column.numeric_column(header)
    feature_columns.append(feature_column.bucketized_column(temp, boundaries=[-89, -70, -65, -60, -55, -50, -40, -30, -20]))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

new_model = tf.keras.models.load_model('guesser.h5',custom_objects={'DenseFeatures':feature_layer})

new_model.summary()

# feature cols
for header in headers:
    temp = feature_column.numeric_column(header)
    feature_columns.append(feature_column.bucketized_column(temp, boundaries=[-89, -70, -65, -60, -55, -50, -40, -30, -20]))
    #feature_columns.append(temp) #old code used without buckets. only numeric columns


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(3, activation='sigmoid') #cambiare con il numero di zone
])

model.save('output_path', format='tf')