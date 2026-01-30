import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow import feature_column
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
df = pd.read_csv(URL)
df.head()

countries = ['afghanistan', 'aland islands', 'albania', 'algeria', 'american samoa', 'andorra', 'angola', 'anguilla', 'antarctica', 'antigua and barbuda', 'argentina', 'armenia', 'aruba', 'australia', 'austria', 'azerbaijan', 'bahamas (the)', 'bahrain', 'bangladesh', 'barbados', 'belarus', 'belgium', 'belize', 'benin', 'bermuda']

df['country'] = pd.DataFrame(np.random.choice(list(countries), len(df)))

feature_columns = []
for f in list(df.columns):
    if f!='target':
        if df[f].dtype.name in ['int64','float64']:
            num_feat = feature_column.numeric_column(f)
            bucket_feat = feature_column.bucketized_column(num_feat, boundaries=[25,50,75,90,95,99])
            feature_columns.append(bucket_feat)
        else:
            categ_feat = feature_column.categorical_column_with_vocabulary_list(f, df[f].unique())
            categ_feat_embedding = feature_column.embedding_column(categ_feat, dimension=8)
            feature_columns.append(categ_feat_embedding)

train_df, val_df = train_test_split(df, test_size=0.2)

batch_size = 128
train_ds = df_to_dataset(train_df, batch_size=batch_size)
val_ds = df_to_dataset(val_df, shuffle=False, batch_size=batch_size)

feature_layer = keras.layers.DenseFeatures(feature_columns)
model = keras.Sequential()
model.add(feature_layer)
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
              loss=keras.losses.BinaryCrossentropy())


history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=2,
                    verbose=1)


model2 = tf.keras.models.model_from_json(model.to_json())
model2.set_weights(model.get_weights())