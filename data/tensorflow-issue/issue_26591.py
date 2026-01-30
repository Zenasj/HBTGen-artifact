import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow.train import Example, Features, Feature, FloatList
import numpy as np

# Create a simple TFRecord file
ages = np.random.rand(100)*50 + 20
heights = np.random.rand(100)*40 + 150
weights = 0.75 * heights - 75. + ages * 0.1 + np.random.rand(100) * 20
with tf.io.TFRecordWriter("weights.tfrecord") as f:
    for ex_age, ex_height, ex_weight in zip(ages, heights, weights):
        example = Example(features=Features(feature={
            "age": Feature(float_list=FloatList(value=[ex_age])),
            "height": Feature(float_list=FloatList(value=[ex_height])),
            "weight": Feature(float_list=FloatList(value=[ex_weight]))
        }))
        f.write(example.SerializeToString())

# Create a TFRecordDataset to read the data
age = tf.feature_column.numeric_column("age")
height = tf.feature_column.numeric_column("height")
weight = tf.feature_column.numeric_column("weight")
columns = [age, height, weight]
feature_descriptions = tf.feature_column.make_parse_example_spec(columns)
def parse_examples(serialized_examples):
    features = tf.io.parse_example(serialized_examples,
                                   feature_descriptions)
    targets = features.pop("weight")
    return features, targets
dataset = tf.data.TFRecordDataset(["weights.tfrecord"])
dataset = dataset.shuffle(100).batch(32).map(parse_examples)

# Create, train and use the model with a DenseFeatures layer
model = keras.models.Sequential([
    keras.layers.DenseFeatures(columns[:-1]),
    keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-5))
history = model.fit(dataset, epochs=5)
y_pred = model.predict({"age": tf.constant([25.]),
                        "height": tf.constant([180.])})

# Saving using the save() method works fine
model.save("my_weight_model.h5")

# Saving to a SavedModel fails
tf.saved_model.save(model, "my_weight_model.savedmodel") # AttributeError!