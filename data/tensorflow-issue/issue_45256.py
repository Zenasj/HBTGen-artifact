import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

model = tf.keras.applications.mobilenet.MobileNet(weights=None, input_shape=(96, 96, 1), alpha = 0.25, classes=6)

converter = tf.lite.TFLiteConverter.from_saved_model("...")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # removing these lines does not fix the error
converter.inference_input_type = tf.int8 # removing these lines does not fix the error
converter.inference_output_type = tf.int8 # removing these lines does not fix the error
tflite_model = converter.convert()
open("model_out.tflite", "wb").write(tflite_model)

tf.keras.applications.MobileNet

classes

input = Input(shape=(window_size, 3, ))
x = Conv1D(8, 3, activation='relu', padding='same')(input)
x = AveragePooling1D(2)(x)
x = Conv1D(16, 3, activation='relu', padding='same')(x)
x = AveragePooling1D(2)(x)
x = Conv1D(32, 3, activation='relu', padding='same')(x)
x = AveragePooling1D(2)(x)
x = Conv1D(32, 3, activation='relu', padding='same')(x)
x = Conv1D(64, 3, activation='relu', padding='same')(x)
#x = GlobalAveragePooling1D()(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
type_out = Dense(len(labels), activation="softmax", name="type")(x)
precip_out = Dense(1, activation="sigmoid", name="precip")(x)

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Softmax, Dropout

model = MobileNet(include_top = False, weights=None, input_shape=(96, 96, 1), pooling='avg', alpha=0.25, dropout=0.001 )
top_layer = Dropout(0.001)(model.output)
top_layer = Dense(2)(top_layer)
top_layer = Softmax()(top_layer)
model = Model(model.input, top_layer)