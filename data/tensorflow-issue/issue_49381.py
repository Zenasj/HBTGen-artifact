import numpy as np
import tensorflow as tf

model = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=( 1, 3,227,227)),
  tf.keras.layers.Conv2D(96, 11, activation='relu', strides=(4,4), dilation_rate=(1,1), groups=1, data_format = 'channels_first')])
model.save('my_conv_model')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()

model.add(layers.LSTM(input_shape=(None,20),units= 128, return_sequences = True))
model.add(layers.Dense(10))
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']);
x_test = np.zeros([3000,4,20], dtype=np.float32);
x_train = np.zeros([3000,4,20], dtype=np.float32);
y_test =  np.zeros([3000, 4,], dtype=np.float32);
y_train = np.zeros([3000, 4,], dtype=np.float32);

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

run_model = tf.function(lambda x: model(x))

BATCH_SIZE = 1
STEPS = 2
INPUT_SIZE = 20

concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))


MODEL_DIR = "keras_lstm_new"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()

with open('lstmmodelnew.tflite', 'wb') as f:
  f.write(tflite_model)