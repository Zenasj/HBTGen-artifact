import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(5, activation="softmax")
])

model.build(input_shape=(1, 224, 449, 3))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("converted_model.tflite", "wb") as f:
    f.write(tflite_model)

class WrappedModel(tf.keras.Model):
    def __init__(self, baseModel):
        super().__init__()
        self.baseModel = baseModel

    def call(self, x, training=False):
        x = tf.reshape(x, [1, 224, 449, 3])
        return self.baseModel(x)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(5, activation="softmax")
])

model.build(input_shape=(1, 224, 449, 3))

wrapped_model = WrappedModel(model)

# Calling wrapped_model.build(...) doesn't work here for some reason, looks like another bug.
wrapped_model.compute_output_shape(
    input_shape=[1, 224 * 449 * 3]
)

converter = tf.lite.TFLiteConverter.from_keras_model(wrapped_model)
tflite_model = converter.convert()
with open("converted_model.tflite", "wb") as f:
    f.write(tflite_model)