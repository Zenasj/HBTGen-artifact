from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# without the following two lines, it will throw
# ValueError: Cannot set tensor: Got value of type NOTYPE but expected type FLOAT32 for input 0, name: flatten_input 
#x_train = tf.dtypes.cast(x_train,tf.float32)
#x_test = tf.dtypes.cast(x_test,tf.float32)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
#converter.experimental_enable_mlir_converter = True
tflite_model = converter.convert()

import numpy as np
expected = model.predict(x_test[0:1])

# Run the model with TensorFlow Lite
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]["index"], x_test[0:1, :, :])
interpreter.invoke()
result = interpreter.get_tensor(output_details[0]["index"])

# Assert if the result of TFLite model is consistent with the TF model.
np.testing.assert_almost_equal(expected, result)
print("Done. The result of TensorFlow matches the result of TensorFLow Lite.")