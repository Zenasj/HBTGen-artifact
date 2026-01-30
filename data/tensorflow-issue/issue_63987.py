import random
from tensorflow.keras import layers
from tensorflow.keras import models

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)

tf.saved_model.save(best_model,save_path)

converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
tflite_model = converter.convert()

#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, DepthwiseConv2D, Conv2D, Flatten, Dense

# Define a simple model with depthwise convolutions
model = Sequential()
model.add(Input(shape=(64, 64, 3)))
model.add(DepthwiseConv2D(kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(64, (1, 1), activation="relu"))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# Define the representative dataset generator
def representative_dataset_gen():
    for _ in range(100):
        # Here you should provide a sample from your actual dataset
        # For illustration, we'll use random data
        # Yield a batch of input data (in this case, a single sample)
        yield [tf.random.normal((1, 64, 64, 3))]


# Set the TensorFlow Lite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the optimization to default for int8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Define the representative dataset for quantization
converter.representative_dataset = representative_dataset_gen

# Restrict the target spec to int8 for full integer quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Instruct the converter to make the input and output layer as integer
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert the model
tflite_model = converter.convert()

# Save the model to a file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model has been successfully converted to TFLite and saved as 'model.tflite'.")