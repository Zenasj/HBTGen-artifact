import numpy as np
import tensorflow as tf

interpreter = tf.contrib.lite.Interpreter(model_path="saved_model.tflite")
input_details = interpreter.get_input_details()
# [{'quantization': (0.0, 0), 'index': 216, 'name': 'input_img', 'shape': array([   1,  704, 1280,    3], dtype=int32), 'dtype': <class 'numpy.float32'>}]

### Resize the input tensor ###
tensor_size = np.array((1,1408,2560,3), dtype=np.int32)
interpreter.resize_tensor_input(input_details[0]['index'],tensor_size)
print(input_details[0]['shape']) 
#  [   1  704 1280    3]