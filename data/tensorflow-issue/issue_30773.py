from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

#Import basics and check everything works
import tensorflow as tf
from tensorflow import keras

AUTOTUNE = tf.data.experimental.AUTOTUNE

print("Versions:", tf.version.VERSION, tf.version.GIT_VERSION)
print("GPU availablilty:", tf.test.is_gpu_available())
print("Eager execution:", tf.executing_eagerly())

#Quick test
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

def make_model(input_shape, n_hidden1=2049, n_hidden2=500, n_hidden3=180, batch_n_mom=0.99, dropout_rate=0.1):

    from tensorflow.keras.initializers import he_normal
    
    stacked_ae = keras.models.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.BatchNormalization(axis=1, momentum=batch_n_mom),
        
        keras.layers.Dense(n_hidden1, activation="selu", name="he1", kernel_initializer=he_normal(seed=27)),
        keras.layers.BatchNormalization(axis=1, momentum=batch_n_mom),
        keras.layers.Dropout(dropout_rate),
        
        keras.layers.Dense(n_hidden2, activation="selu", name="he2", kernel_initializer=he_normal(seed=42)),
        keras.layers.BatchNormalization(axis=1, momentum=batch_n_mom),
        
        keras.layers.Dense(n_hidden3, activation="selu", name="he3", kernel_initializer=he_normal(seed=65)),
        keras.layers.BatchNormalization(axis=1, momentum=batch_n_mom),
        
        keras.layers.Dense(n_hidden2, activation="selu", name="hd2", kernel_initializer=he_normal(seed=42)),
        keras.layers.BatchNormalization(axis=1, momentum=batch_n_mom),
        
        keras.layers.Dense(n_hidden1, activation="selu", name="hd1", kernel_initializer=he_normal(seed=27)),
        keras.layers.BatchNormalization(axis=1, momentum=batch_n_mom),
        keras.layers.Dropout(dropout_rate),
        
        keras.layers.Dense(input_shape[0] * input_shape[1], name="output", kernel_initializer=he_normal(seed=62)),
        keras.layers.Reshape(input_shape)
    ])
    
    return stacked_ae

import numpy as np

#Data doesn't matter
x_train = np.ones((32,60,80))
y_train = np.ones((32,60,80))

#Once runs ok
input_shape = [60,80]
ae_model = make_model(input_shape)
ae_model.compile(loss="mse",
                 optimizer=keras.optimizers.Adam(learning_rate=0.001, decay=1e-6),
                metrics=['accuracy'])
print(ae_model.summary())

#Do something with the model
history = ae_model.fit(x=x_train, y=y_train,  epochs=1, steps_per_epoch=1)

#Second run, new model
ae_model = make_model(input_shape, n_hidden1=2150)
ae_model.compile(loss="mse",
                 optimizer=keras.optimizers.Adam(learning_rate=0.001, decay=1e-6),
                metrics=['accuracy'])
print(ae_model.summary())

#Run again. GPU OOM.
history = ae_model.fit(x=x_train, y=y_train,  epochs=1, steps_per_epoch=1)

batch_size

tf.config.experimental.set_memory_growth