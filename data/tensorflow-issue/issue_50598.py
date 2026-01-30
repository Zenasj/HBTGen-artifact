import numpy as np
import tensorflow as tf
from tensorflow.keras import models

model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(n_timesteps, n_features))) # This layer is used in the final model
model.add(LSTM(64, input_shape=(n_timesteps,n_features))) # This layer is used in the final model
model.add(RepeatVector(n_timesteps))
model.add(LSTM(64, return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
optimizer = Adam(learning_rate=0.001, epsilon=1e-04)
model.compile(optimizer= optimizer, loss='mse')
model.fit(x_train, x_train, epochs=1000, verbose=2)

# Saving
encoder = Model(inputs=model.inputs, outputs=model.layers[1].output)
encoder.save('encoder.h5')

from tensorflow.keras.models import load_model
encoder = load_model('encoder.h5')

# Following code from https://www.tensorflow.org/lite/convert/rnn
run_model = tf.function(lambda x: encoder(x))
BATCH_SIZE = None
STEPS = 6952
INPUT_SIZE = 20
concrete_func = run_model.get_concrete_function(tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], encoder.inputs[0].dtype))
encoder.save('/encoder', save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model('/encoder')
converter.experimental_new_converter = True
tflite_model = converter.convert()
with open('encoder.tflite', 'wb') as f:
  f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path='encoder.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
data = np.vstack([a,a]).astype(np.float32) # Shape: (2, 6952, 20)
interpreter.resize_tensor_input(input_details[0]['index'], data .shape) # Shape: (2, 6952, 20)
interpreter.resize_tensor_input(output_details[0]['index'], (data .shape[0], 64)) # Shape: (2, 64)
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], data )
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])