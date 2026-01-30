from tensorflow import keras

import tensorflow as tf
import numpy as np

names = tf.keras.Input(shape=(2,), dtype=tf.string)
model = tf.keras.Model(
    inputs=names,
    outputs=tf.gather(names, tf.constant([0])),
)

model.save('./export')
converter = tf.lite.TFLiteConverter.from_saved_model('./export')
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], np.array([[1, 2]], dtype=np.string))
interpreter.invoke()
interpreter.get_tensor(output_details[0]['index'])

import tensorflow as tf
import numpy as np

names = tf.keras.Input(shape=(2,), dtype=tf.string)
model = tf.keras.Model(
    inputs=names,
    outputs=tf.gather(names, tf.constant([0])),
)

model(np.array([1, 2], dtype=np.str))