from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import pathlib

import tensorflow as tf


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  # tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)


# Save the model into SaveModel format

saved_model_dir = pathlib.Path("./saved_model/")
tf.saved_model.save(model, str(saved_model_dir))

import pathlib

import tensorflow as tf


mnist = tf.keras.datasets.mnist
x_train = mnist.load_data()[0][0] / 255.0

saved_model_dir = pathlib.Path("./saved_model/")


# Convert the model from saved model

images = tf.cast(x_train, tf.float32)
mnist_ds = tf.data.Dataset.from_tensor_slices(images).batch(1)


def representative_dataset_gen():
    for input_value in mnist_ds.take(100):
        yield [input_value]


converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen

tflite_quant_model = converter.convert()

tflite_quant_model_file = saved_model_dir/"mnist_post_quant_model_io.tflite"
tflite_quant_model_file.write_bytes(tflite_quant_model)