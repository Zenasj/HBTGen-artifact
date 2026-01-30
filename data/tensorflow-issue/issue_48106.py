import random

import numpy as np
import tensorflow as tf


class Tester(tf.Module):
    def __init__(self):
        super(Tester, self).__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=[100], dtype=tf.float32)])
    def test(self, x):
        # return tf.reshape(x, [10, -1])
        return tf.signal.rfft(x, [512])


model = Tester()
concrete_func = model.test.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir='saved_models/pb/model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite = converter.convert()

TFLITE_FILE_PATH = 'content/tester.tflite'
with open(TFLITE_FILE_PATH, 'wb') as f:
    f.write(tflite)

# Load the TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="content/test_variable.tflite")
interpreter = tf.lite.Interpreter(model_content=tflite)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)