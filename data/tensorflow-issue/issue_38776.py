import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.lite.python import lite
from tensorflow.python import keras
import numpy as np

input_a = keras.layers.Input(shape=(3,3,), name='input_a')
interm_b = tf.keras.layers.LSTM(4, name='interm_1')(input_a)
output_c = keras.layers.Dense(1, name='dense_1')(interm_b)

model = tf.keras.models.Model(inputs=[input_a], outputs=[output_c])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.summary()

batch_size = 10
sample_input = np.ones((batch_size,3,3),dtype=np.float32)

expected_value = model.predict(sample_input)

converter = lite.TFLiteConverterV2.from_keras_model(model = model)
converter.experimental_new_converter = True
with open("model.tflite", "wb") as f:
    f.write(converter.convert())

interpreter = lite.Interpreter(model_path="model.tflite")
print(interpreter.get_input_details())
interpreter.resize_tensor_input(0,[batch_size, 3,3])
interpreter.allocate_tensors()
interpreter.set_tensor(0, sample_input)
interpreter.invoke()
interpreter.get_tensor(interpreter.get_output_details()[0]["index"])