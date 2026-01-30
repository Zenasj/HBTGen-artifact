from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import os

model_dir = "models/feature_column_example"
category = tf.constant(["A", "B", "A", "C", "C", "A"])
label = tf.constant([1, 0, 1, 0, 0, 0])

ds = tf.data.Dataset.from_tensor_slices(({"category": category}, label))
ds = ds.batch(2)

fc_category = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
        "category", vocabulary_list=["A", "B", "C"]
    )
)
feature_layer = tf.keras.layers.DenseFeatures([fc_category])

model = tf.keras.Sequential(
    [
        feature_layer,
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

model.fit(ds, epochs=2)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.allow_custom_ops = True
# converter.experimental_new_converter = True
# converter.experimental_new_quantizer = True

# Convert the model.
tflite_model = converter.convert()
open(os.path.join(model_dir, "output.tflite"), "wb").write(tflite_model)