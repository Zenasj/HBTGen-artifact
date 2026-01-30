from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

print("TF version:", tf.__version__)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

tensor_details = interpreter.get_tensor_details()
print("Found", len(tensor_details), "tensors:")
for tensor in tensor_details:
    print(f"Tensor {tensor['index']}: {tensor['name']}")
    numpy_tensor = interpreter.get_tensor(tensor["index"])
    print("numpy shape:", numpy_tensor.shape)