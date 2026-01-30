from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

input_data = Input(shape=(5,))
model = Model(inputs=input_data, outputs=input_data)
model.compile(loss='mae')

class valCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, inputs, outputs):
        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        
    def on_epoch_end(self, epoch, logs={}):
        val = self.model.evaluate(self.inputs, self.outputs, verbose=0)
        print("\nVAL: ", val)

traindata = np.ones((5))
vc = valCallback(model, traindata, 3*traindata)  
model.fit(traindata, traindata, 
          validation_data=(traindata, 2*traindata),
          epochs=4, callbacks=[vc], validation_freq=2, verbose=0)