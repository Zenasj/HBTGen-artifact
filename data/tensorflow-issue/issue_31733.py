import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)

class FeatureColumnDsModel(tf.keras.models.Model):
    def __init__(self, feature_columns, **kwargs):
        super().__init__(name=None, **kwargs)
        self._input_layer = tf.keras.layers.DenseFeatures(feature_columns)
        self._dense1 = tf.keras.layers.Dense(128, activation='relu')
        self._dense2 = tf.keras.layers.Dense(128, activation='relu')
        self._output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, features):
        x = self._input_layer(features)
        x = self._dense1(x)
        x = self._dense2(x)
        y = self._output_layer(x)
        return y
    
model = FeatureColumnDsModel(feature_columns)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'],)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)