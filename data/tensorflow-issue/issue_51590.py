from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
import pathlib


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device = gpu, enable = True)


# Set keras model name
keras_model = "weight.h5"
        
# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Define the model architecture
model = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=(28, 28)),
  tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
  tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10)
])


# Set training details
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model
model.fit(
  train_images,
  train_labels,
  epochs=5,
  validation_data=(test_images, test_labels)
)

# Save model
model.save_weights(filepath = keras_model, save_format = 'h5')

# Load keras model weight
model.load_weights(filepath = keras_model)


# Define representative dataset
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    yield [input_value]


# Do conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()   # FAILED HERE


# Save the quantized model
tflite_models_dir = pathlib.Path("")
tflite_model_quant_file = tflite_models_dir/tflite_model
tflite_model_quant_file.write_bytes(tflite_model_quant)

import tensorflow as tf
assert tf.version.VERSION == "2.5.0", "Please install TF 2.6.0, you're currently using TF " + tf.version.VERSION