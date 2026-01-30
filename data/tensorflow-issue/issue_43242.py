import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

n_epoch = 1
n_batch = 1
n_neurons= 25
time_ev = 6

# design network
model = Sequential()
model.add(tf.keras.layers.Input(batch_input_shape=(n_batch,time_ev, 3), name='input')) # nbatch, num_feature, time
model.add(tf.keras.layers.Flatten())
model.add(Dense(1, activation=tf.keras.activations.relu))
model.compile(loss='mean_squared_error', optimizer='adam')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()