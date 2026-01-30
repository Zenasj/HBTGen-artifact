import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

reallocate_tensors = False  # Turn this on to reallocate tensors every run, fixing tflite accuracy
nb_channels = 5
wt_size = 100
in_size = 30

class DynamicWeightTest(tf.keras.layers.Layer):
    def __init__(self,
                 channels,
                 select_size,
                 **kwargs
                 ):
        super().__init__()

        self.channels = channels
        self.select_size = select_size


    def call(self, inputs):
        x = inputs[0]
        wt = inputs[1]

        x = tf.nn.conv2d(x, wt, (1, 1), 'VALID')

        return x


input_wt = tf.keras.layers.Input(shape=(1, nb_channels, wt_size), dtype=tf.float32)
input_data = tf.keras.layers.Input(shape=(in_size, in_size, nb_channels,), dtype=tf.float32)
x = DynamicWeightTest(channels=wt_size, select_size=nb_channels)([input_data, input_wt])

model = tf.keras.Model(inputs=[input_data, input_wt], outputs=[x])

# Get the concrete function from the Keras model.
model_fn = tf.function(lambda x: model(x))
model_fn_concrete = model_fn.get_concrete_function([input_data, input_wt])


# tflite
converter = tf.lite.TFLiteConverter.from_concrete_functions([model_fn_concrete])
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for _ in range(10):
    data = np.array(np.random.random_sample((1, in_size, in_size, nb_channels,)), dtype=np.float32)
    wt = np.array(np.random.random_sample((1, 1, nb_channels, wt_size)), dtype=np.float32)
    out_ref = model([data, wt])

    # concrete
    out_fn = model_fn_concrete(tf.constant(data), tf.constant(wt))

    # Note: need to fake resize the input & reallocate tensors
    if reallocate_tensors:
        interpreter.resize_tensor_input(0, (1, in_size, in_size, nb_channels,))
        interpreter.allocate_tensors()

    # Call tflite
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.set_tensor(input_details[1]['index'], wt)
    interpreter.invoke()
    out_test = interpreter.get_tensor(output_details[0]['index'])

    out_ref = out_ref.numpy()
    out_fn = out_fn.numpy()
    diff = 100. * np.sum(np.abs(out_test - out_ref)) / np.sum(np.abs(out_ref))
    diff_fn = 100. * np.sum(np.abs(out_fn - out_ref)) / np.sum(np.abs(out_ref))
    test_sum = np.sum(out_test)

    print('Diff concrete function is: %f, diff tflite is: %f' % (diff_fn, diff))