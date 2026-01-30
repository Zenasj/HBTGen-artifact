from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


data = [{"a": 20.0, "b": 5.0, "c": 0.2972786923202068}, {"a": 20.0, "b": 10.0, "c": 0.10673704592967688}]
train_dataset = pd.DataFrame.from_dict(data)

train_features = train_dataset.copy()
train_labels = train_features.pop('c')

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

dnn_model = keras.Sequential([
    normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

dnn_model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
dnn_model.save('file.h5')


# ValueError: Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor.
new_model = tf.keras.models.load_model('file.h5')

json_config = dnn_model.to_json()
new_model = keras.models.model_from_json(json_config)