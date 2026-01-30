from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf


input = tf.keras.Input(shape=(1))
output = tf.keras.layers.Multiply()([input, input])
model = tf.keras.Model(inputs=input, outputs=output)


def representative_data_gen():
    yield [np.array([[1.3]], dtype=np.float32)]
    yield [np.array([[1.4]], dtype=np.float32)]
    # It works with this line if _experimental_new_quantizer=True
    # yield [np.array([[0.0]], dtype=np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.representative_dataset = representative_data_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter._experimental_new_quantizer = True

tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input = np.array([[1.4]], dtype=np.float32)
interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input)
interpreter.invoke()
output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])

print("input ", input)
print("output ", output)