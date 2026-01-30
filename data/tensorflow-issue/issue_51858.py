from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

py
import tensorflow as tf

print(tf.version.VERSION)

input = tf.keras.layers.Input(shape=(3,3,32), name="input")

o1 = tf.keras.layers.Conv2D(2, (1,1), activation='relu', input_shape=(1,3,3,32))(input)
o2 = tf.keras.layers.Conv2D(16, (1,1), activation='relu', input_shape=(1,3,3,32))(input)
o3 = tf.keras.layers.Conv2D(32, (1,1), activation='relu', input_shape=(1,3,3,32))(input)

model = tf.keras.Model(inputs=input, outputs=[o1,o2,o3])
model.summary()

tf.keras.models.save_model(model, "test_saved_model")

py
import tensorflow as tf

print(tf.version.VERSION)

converter = tf.lite.TFLiteConverter.from_saved_model("test_saved_model")
converter.allow_custom_ops = True
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()
open("test_saved_model.tflite", "wb").write(tflite_model)

import tensorflow as tf

print(tf.version.VERSION)

input = tf.keras.layers.Input(shape=(3,3,32), name="input")

o1 = tf.keras.layers.Conv2D(2, (1,1), activation='relu', input_shape=(1,3,3,32))(input)
o2 = tf.keras.layers.Conv2D(16, (1,1), activation='relu', input_shape=(1,3,3,32))(input)
o3 = tf.keras.layers.Conv2D(32, (1,1), activation='relu', input_shape=(1,3,3,32))(input)

model = tf.keras.Model(inputs=input, outputs=[o1,o2,o3])
print(model.summary())

tf.keras.models.save_model(model, "test_saved_model")
converter = tf.lite.TFLiteConverter.from_saved_model("test_saved_model")
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
print(interpreter.get_signature_list())
output_details = interpreter.get_output_details()
print(output_details)