from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import DenseFeatures, Dense, Input

def make_model(features):
    feature_columns = [tf.feature_column.numeric_column(key) for key in features]
    nn_input = {key: Input(name=key, shape=(), dtype=tf.float32) for key in features}

    feat = DenseFeatures(feature_columns)(nn_input)
    dense = Dense(16)(feat)
    output = Dense(1)(dense)
    model = tf.keras.Model(inputs=nn_input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.AUC()],
    )
    return model

features = ["age", "income"]
label = "is_male"

input_dataset = tf.data.Dataset.from_tensor_slices(
    {key: np.ones((1000, 1), dtype=np.float) for key in features}
)
target_dataset = tf.data.Dataset.from_tensor_slices(
    {label: np.ones((1000, 1), dtype=np.int)}
)
complete_dataset = tf.data.Dataset.zip((input_dataset, target_dataset)).shuffle(10000)

model = make_model(features)
model.summary()
model.fit(complete_dataset)