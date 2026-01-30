from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

#!/usr/bin/env python3.8
import numpy as np
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
SEED = 1000
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.random.set_seed(SEED)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

@tf.keras.utils.register_keras_serializable()
class Custom_Layer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(Custom_Layer, self).__init__(**kwargs)
        self.units = units
    def call(self, x):
        x = tf.keras.layers.Dense(self.units)(x)
        return x
    def get_config(self):
        config = super(Custom_Layer, self).get_config()
        config.update(units = self.units)
        return config

def build_model(input_shape, units):
    inputs = tf.keras.Input(shape=input_shape)
    x = Custom_Layer(units)(inputs) # problem is here
    # x = tf.keras.layers.Dense(units)(inputs) # if replace the above line with this line, can give the same result
    x = tf.keras.layers.Dense(1)(x)
    outputs = x[:,-1,:]
    return tf.keras.Model(inputs, outputs)

x_train = np.random.random((20000, 10, 50))
y_train = np.random.random(20000)
model = build_model(input_shape=x_train.shape[1:], units=256)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0))
hist = model.fit(x_train, y_train, epochs=1)

x_test = np.random.random((5, 10, 50))
predicted_val = model.predict(x_test)
print (predicted_val)
h5_file_name = '/tmp/model.h5'
model.save(h5_file_name)

model2 = tf.keras.models.load_model(h5_file_name, custom_objects={'Custom_Layer':Custom_Layer})
predicted_val2 = model2.predict(x_test)
print (predicted_val2)