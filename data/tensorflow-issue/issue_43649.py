import random
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape):
    X_input = layers.Input(shape = input_shape)

    X = layers.Conv1D(128, 7, strides=4)(X_input)                           
    X = layers.BatchNormalization()(X)                                 
    X = layers.Activation('relu')(X)                                 
    X = layers.Dropout(0.5)(X)                                 

    X = layers.LSTM(64, return_sequences=True)(X)                                
    X = layers.Dropout(0.5)(X)                           
    X = layers.BatchNormalization()(X)                                
    
    X = layers.LSTM(64, return_sequences=True)(X)                              
    X = layers.Dropout(0.5)(X)                                
    X = layers.BatchNormalization()(X)                                
    X = layers.Dropout(0.5)(X)

    X = layers.TimeDistributed(layers.Dense(1, activation = "sigmoid"))(X) 

    model = keras.models.Model(inputs = X_input, outputs = X)
    return model

model = build_model((1000, 100))
model.summary()

x = np.random.rand(10, 1000, 100).astype(np.float32)
y = np.random.randint(2, size=(10, 249)).astype(np.float32)

model.compile('adam', 'binary_crossentropy')
model.fit(x, y, epochs=1)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()