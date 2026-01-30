import math
from tensorflow import keras

import tensorflow as tf
import numpy as np

input_data = np.array([-0.2, -0.33, -1.0, -0.4, -0.1], dtype=np.float32)

inp = tf.keras.Input(shape=(5,))
out = tf.math.abs(inp)
model = tf.keras.Model(inp, out)

# Post training quantization
def representative_dataset():
    yield [input_data]
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_quant_model = converter.convert()
tflite_model_path = "model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_quant_model)

# Tflite inference with the quantized model
interpreter = tf.lite.Interpreter(tflite_model_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]["index"], [input_data])
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]["index"])

# The quantized model doesn't behave as expected:
# The quant output should be able to represent positive numbers
# However, the quant output can't represent positive numbers
# Input_scale = output_scale = 0.003921568859368563
# Input_zero_point = output_zero_point = -127
print("input: ", input_data)  # [-0.2, -0.33, -1.0, -0.4, -0.1]
print("expected output: ", model(input_data))  # [0.2  0.33 1.   0.4  0.1 ]
print("quantized output: ", output_data)  # [[0. 0. 0. 0. 0.]]