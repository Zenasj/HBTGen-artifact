from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_data():
    images = np.zeros((64,224))
    labels = np.zeros((64,5))
    inputs = {
        'Image_input':images
    }
    outputs = {
        'output-softmax':labels
    }
    return inputs, outputs

def create_model():
    input_layer = tf.keras.layers.Input(name='Image_input', shape=(224), dtype='float32')
    model = tf.keras.layers.Dense(5)(input_layer)
    model = tf.keras.layers.Activation('softmax', name = "output-softmax")(model)
    model = tf.keras.models.Model(inputs=input_layer, outputs=[model])
    return model
    
model = create_model()

optimizer = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    
data = get_data()
model.fit(data[0], data[1], batch_size=16, epochs=10)

tf.keras.models.save_model(model,"model")

model1 = tf.keras.models.load_model("model")

model1.fit(data[0], data[1], batch_size=16, epochs=10)

def get_data():
        images = np.zeros((64,224))
        labels = np.zeros((64,5))
        return images,labels