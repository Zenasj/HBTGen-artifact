import random
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
import numpy as np

class LogisticEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = keras.metrics.BinaryAccuracy()

    def call(self, targets, logits, sample_weights=None):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # Log accuracy as a metric and add it
        # to the layer using `self.add_metric()`.
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="accuracy")

        # Return the inference-time prediction tensor (for `.predict()`).
        return tf.nn.softmax(logits)

inputs = keras.Input(shape=(3,), name="inputs")
targets = keras.Input(shape=(10,), name="targets")
logits = keras.layers.Dense(10)(inputs)
predictions = LogisticEndpoint(name="predictions")(logits, targets)

model = keras.Model(inputs=[inputs, targets], outputs=predictions)
model.compile(optimizer="adam")

data = {
    "inputs": np.random.random((3, 3)),
    "targets": np.random.random((3, 10)),
}
model.fit(data)

tf_data = {
    'inputs': tf.data.Dataset.from_tensor_slices(np.random.random((3, 3))), 
    'targets': tf.data.Dataset.from_tensor_slices(np.random.random((3, 10)))
}

model.fit(tf_data)

data = {
    "inputs": np.random.random((3, 3)),
    "targets": np.random.random((3, 10)),
}
tuple_data = []
for i in range(3):
  tuple_data.append((data['inputs'][i, :], data['targets'][i, :]))

dataset = tf.data.Dataset.from_generator(lambda: tuple_data, (tf.float32, tf.float32))

# to show the dataset content
for x in dataset:
  print(x)

model.fit(dataset)

input = tf.random.uniform([3, 3])
targets = tf.random.uniform([3, 10])
tf_data = tf.data.Dataset.from_tensor_slices({'inputs': input, 'targets': targets})
tf_data = tf_data.batch(2)
model.fit(tf_data)