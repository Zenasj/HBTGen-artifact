import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

input0 = keras.Input(shape=(64,), name="input0") 
input1 = keras.Input(shape=(16,), name="input1")
input2 = keras.Input(shape=(16,), name="input2")

layer0 = layers.Dense(3, activation="relu", name="layer0")(input0)
layer1 = layers.Dense(5, activation="relu", name="layer1")(input1)
layer2 = layers.Dense(7, activation="relu", name="layer2")(input2)

x0 = layers.concatenate([layer0, layer1, layer2])
x1 = layers.concatenate([layer0, layer1, layer2])

output0 = layers.Dense(3, name="output0")(x0)
output1 = layers.Dense(7, name="output1")(x1)

model = keras.Model(
    inputs=[input0, input1, input2],
    outputs=[output0, output1],
)
model.compile(optimizer='sgd', loss='mean_squared_error')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)