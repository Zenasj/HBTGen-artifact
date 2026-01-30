from tensorflow import keras
from tensorflow.keras import models

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model

def build_toy():
    
    inputs = layers.Input(shape=(1,), name="inputs")
    x = layers.Dense(1, activation="linear", kernel_initializer=tf.keras.initializers.Constant(value=1), name="dense") (inputs)
    x = layers.Dropout(0.5, seed=12, name="dropout") (x,  training=True)
    out = x
    
    model = Model(inputs, out)
    
    return model

data = np.array(([1]))
    
predict_model = build_toy()
print(predict_model.predict(data))

call_model = build_toy()
print(call_model(data))