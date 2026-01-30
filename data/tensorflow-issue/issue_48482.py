import random
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.layers import concatenate, Input, LSTM, Bidirectional, Embedding, Dense, TimeDistributed,SpatialDropout1D
from tensorflow.keras.models import Model
import tensorflow as tf
print(tf.__version__)

# Create Tensorflow model 
word_in = Input(shape=(300,), name="input_wor")
emb_wor = Embedding(input_dim=1834, output_dim=16, input_length=300, mask_zero=True, name="emb_wor")(word_in)
char_in = Input(shape=(300, 20 ,), name="input_char")
emb_char = TimeDistributed(Embedding(input_dim=132, output_dim=32, input_length=20, mask_zero=True, name="emb_char"))(char_in)
char_enc = TimeDistributed(LSTM(units=32, return_sequences=False, recurrent_dropout=0.15, name="char_enc"))(emb_char)
input_pos = Input(shape=(300, 4, ), name="input_pos")
input_par = Input(shape=(300, 3, ), name="input_par")

x = concatenate([emb_wor, char_enc, input_pos, input_par])
x = SpatialDropout1D(0.1)(x)
main_lstm = Bidirectional(LSTM(units=64, return_sequences=True, dropout=0., recurrent_dropout=0.1, name="main_lstm"))(x)
inputs=[word_in,char_in, input_pos, input_par]
outputs = TimeDistributed(Dense(4, activation="softmax", name="out"))(main_lstm)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
print(model.summary())


# Convert Model to Tensorflow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.experimental_new_converter = True
tflite_model = converter.convert()
with open("model.tflite", 'wb') as f:
  f.write(tflite_model)


# # Install tflite_runtime
# !pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

use_tflite_runtime = False # If True then you need to first restart runtime before running this code
import numpy as np

if(use_tflite_runtime):
  import tflite_runtime.interpreter as tflite
  interpreter =  tflite.Interpreter(model_path="model.tflite")
else:
  import tensorflow as tf
  interpreter = tf.lite.Interpreter(model_path="model.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("input_details", input_details)
print("output_details", output_details)

# Set random values
for i in range(len(input_details)):
    x = np.random.random(input_details[i]["shape"])
    interpreter.set_tensor(i, x.astype(input_details[i]["dtype"]))
    print(i, input_details[i]["name"], input_details[i]["shape"], input_details[i]["dtype"], "/", x.shape)

# Invoke
interpreter.invoke()

converter._experimental_lower_tensor_list_ops = False

import tflite_runtime.interpreter as tflite
interpreter =  tflite.Interpreter(model_path="model.tflite")

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False