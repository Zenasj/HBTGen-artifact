import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

class MyModel(tf.keras.Model):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def serve(self, data):
        ...

import tensorflow as tf
import numpy as np

class FullyConnectedModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(10)

    def predict_step(self, data):
        logits = self(data, training=False)
        return tf.argmax(logits, axis=-1)

    def call(self, inputs):
        return self.dense(inputs)

x, y = np.random.uniform(size=(128, 20)).astype(np.float32), np.random.randint(0, 10, size=(128))

model = FullyConnectedModel()
model.compile(optimizer="sgd", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(x, y, epochs=2, batch_size=32)

model.save("/tmp/model", save_traces=True)
reloaded_model = tf.keras.models.load_model("/tmp/model")

y_pred = model.predict(x)
reloaded_y_pred = reloaded_model.predict(x)

np.testing.assert_allclose(reloaded_y_pred, y_pred)

print(y_test.shape, y_test.dtype)
print(y_pred.shape, y_pred.dtype)

class ServingFullyConnectedModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(10)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 20], dtype=tf.float32)])
    def serve(self, data):
        logits = self(data, training=False)
        return tf.argmax(logits, axis=-1)

    def call(self, inputs):
        return self.dense(inputs)