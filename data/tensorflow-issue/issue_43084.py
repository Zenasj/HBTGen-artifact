import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(batch_input_shape=(1,6, 3), name='input'),
    tf.keras.layers.LSTM(n_neurons, time_major=False, return_sequences=False),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name='output')
])
model.compile(loss='mean_squared_error', optimizer='adam')

run_model = tf.function(lambda x: model(x))
# This is important, let's fix the input size.
BATCH_SIZE = 1
STEPS = time_ev
INPUT_SIZE = 3
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))

# model directory.
MODEL_DIR = "keras_lstm"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)