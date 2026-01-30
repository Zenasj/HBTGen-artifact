import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

converter = tf.lite.TFLiteConverter.from_saved_model("<model_folder>")
converter.experimental_new_converter = True
tfmodel = converter.convert()
open(args.output , "wb").write(tfmodel)
interpreter= tf.lite.Interpreter("<tflite_model_name>")
print(interpreter.get_input_details())

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import concatenate

vocab_size = 10
max_length = 10
def create_model():
    input_str = Input(shape=(max_length,))
    x = Embedding(vocab_size, 256, mask_zero=True)(input_str)
    x = LSTM(256)(x)
    input_int = Input(shape=(2))
    y = input_int
    z = concatenate([y, x])
    outputs = Dense(vocab_size, activation='softmax')(z)
    return Model([input_int, input_str], outputs, name='lstm')

model = create_model()
model.summary()
model.save("model_lstm2")

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import concatenate
vocab_size = 10
max_length = 10
def create_model():
    input_str = Input(shape=(max_length), batch_size=1)
    x = Embedding(vocab_size, 256, mask_zero=True)(input_str)
    x = LSTM(256)(x)
    input_int = Input(shape=(2), batch_size=1)
    y = input_int
    z = concatenate([y, x])
    outputs = Dense(vocab_size, activation='softmax')(z)
    return Model([input_int, input_str], outputs, name='lstm')

model = create_model()
model.summary()
model.save("model_lstm2")

run_model = tf.function(lambda x: model(x))
BATCH_SIZE = 1
STEPS = 1
INPUT_SIZE = 2
concrete_func = run_model.get_concrete_function(tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))
model.save("model_lstm2", save_format="tf", signatures=concrete_func)

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import concatenate
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

vocab_size = 10
max_length = 10
def create_model():
    input_str = Input(shape=(max_length,))
    x = Embedding(vocab_size, 256, mask_zero=True)(input_str)
    x = LSTM(256)(x)
    input_int = Input(shape=(2))
    y = input_int
    z = concatenate([y, x])
    outputs = Dense(vocab_size, activation='softmax')(z)
    return Model([input_int, input_str], outputs, name='lstm')

model = create_model()
model.summary()
model.save("model_lstm2")
run_model = tf.function(lambda x, y: model([x, y]))
BATCH_SIZE = 1
STEPS = 1
INPUT_SIZE = 2
concrete_func = run_model.get_concrete_function(tf.TensorSpec((BATCH_SIZE, 2), model.inputs[0].dtype),tf.TensorSpec((BATCH_SIZE, 10), model.inputs[0].dtype))
model.save("model_lstm2", save_format="tf", signatures=concrete_func)