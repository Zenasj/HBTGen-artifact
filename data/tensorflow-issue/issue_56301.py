from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import os
import numpy as np
import tensorflow as tf


def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(X_train.shape[0]):
    # Model has only one input so each data point has one element.
        yield [input_value]


training = ""
model_save_path = ""

X_train = np.loadtxt(training, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 3 * 2 * 21) + 1)))
y_train = np.loadtxt(training, delimiter=',', dtype='int32', usecols=(0))
    
model = tf.keras.models.Sequential([                            
          tf.keras.layers.InputLayer(input_shape=(21 * 3 * 2 * 21, )),
          tf.keras.layers.Reshape((21, 3 * 2 * 21)),        
          tf.keras.layers.LSTM(8, return_sequences=True, unroll=True),
          tf.keras.layers.Dropout(0.40, seed=42),
          tf.keras.layers.LSTM(8, unroll=True),        
          tf.keras.layers.Dropout(0.50, seed=42),             
          tf.keras.layers.Dense(27, activation='softmax')])    
               
           
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=256,
)

model.save(model_save_path, include_optimizer=False)   
model = tf.keras.models.load_model(model_save_path)

run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, model.inputs[0].shape[1]], model.inputs[0].dtype))

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])  
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
#converter.representative_dataset = representative_data_gen
converter.experimental_new_quantizer = True
converter.experimental_enable_resource_variables = True
converter.allow_custom_ops = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_quantized_model = converter.convert()


open("", 'wb').write(tflite_quantized_model)

interpreter = tf.lite.Interpreter(model_path="")
interpreter.allocate_tensors()