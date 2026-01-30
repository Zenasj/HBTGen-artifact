from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

input1_shape = (2,)
# Doesn't work (broadcasting)
input2_shape = (1,)
# Works (no broadcasting)
# input2_shape = (2,)

def get_dequantized_value(quantized_val, params):
    quant_params = params["quantization_parameters"]
    scale = quant_params["scales"]
    zero_point = quant_params["zero_points"]

    return (quantized_val - zero_point) * scale

# Model
input1 = tf.keras.Input(shape=input1_shape)
input2 = tf.keras.Input(shape=input2_shape)
output = tf.keras.layers.Multiply()([input1, input2])
model = tf.keras.Model(inputs=[input1, input2], outputs=output)
model.save("model.h5")

# Convert to TFLite
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("model.h5")
converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0]: (0.0, 3.0),
                                   input_arrays[1]: (0.0, 3.0)}
converter.default_ranges_stats = (0, 9.0)
tflite_model = converter.convert()

# Test TFLite
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input1 = np.full(shape=(1, *input1_shape), fill_value=1, dtype=np.uint8)
interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input1)
print("input1: ", get_dequantized_value(input1, interpreter.get_input_details()[0]))

input2 = np.full(shape=(1, *input2_shape), fill_value=1, dtype=np.uint8)
interpreter.set_tensor(interpreter.get_input_details()[1]["index"], input2)
print("input2: ", get_dequantized_value(input2, interpreter.get_input_details()[1]))

interpreter.invoke()

output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
print("output: ", get_dequantized_value(output, interpreter.get_output_details()[0]))