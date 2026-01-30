from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

inputs = tf.keras.Input(shape=(512, 512, 1))
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                           activation=None, padding='same', use_bias=False)(inputs)
output = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3),
                                         strides=(2, 2), padding='same',
                                         activation=tf.nn.relu)(x)
model = tf.keras.Model(inputs=inputs, outputs=output)

model.save_weights("results/checkpoint")
tf.keras.models.save_model(model, "results/SavedModel", save_format="tf",
                           overwrite=True, include_optimizer=False)
print("Model saved to SavedModel")

converter = trt.TrtGraphConverterV2(input_saved_model_dir="results/SavedModel")
converter.convert()
converter.save("results/trt_model")
print("Model exported to TRT")