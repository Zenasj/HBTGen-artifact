from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
model= tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(None, 32), batch_size=1,name='input'))
model.add(tf.keras.layers.LSTM(256, return_sequences=True, stateful=True))
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['acc'])
print(model.input)
print(model_ctor.summary())
print(tf.__version__)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.experimental_new_converter = True
tflite_model = converter.convert()

# Put link here or attach to the issue.

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(None, 32), batch_size=1,name='input'))
model.add(tf.keras.layers.LSTM(256, return_sequences=True, stateful=True))
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['acc'])
print(model.input)
print(model.summary())
model.save("D:\keras_lstm", save_format='tf')

converter = tf.lite.TFLiteConverter.from_saved_model("D:\keras_lstm")
print(tf.__version__)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
converter.experimental_new_converter = True
tflite_model = converter.convert()

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(None, 32), batch_size=1,name='input'))
model.add(tf.keras.layers.LSTM(256, return_sequences=True, stateful=True))
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['acc'])
print(model.input)
print(model.summary())
model.save("D:\keras_lstm", save_format='tf')

converter = tf.lite.TFLiteConverter.from_saved_model("D:\keras_lstm")
print(tf.__version__)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
converter.experimental_new_converter = True
tflite_model = converter.convert()