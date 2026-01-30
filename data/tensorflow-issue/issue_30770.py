import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

def create_and_train(x_training,y_training,n_cols_in,modelparams):
    layers = [tf.keras.layers.Dense(n_cols_in,activation="relu"),
    tf.keras.layers.Dropout(.5)]
    for param in modelparams:
        layers.extend([tf.keras.layers.Dense(param,activation="sigmoid"),tf.keras.layers.Dropout(.5)])
    layers.append(tf.keras.layers.Dense(1,activation="sigmoid"))
    model = tf.keras.models.Sequential(layers)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    model.fit(x_training, y_training, epochs = epochs)
    with open("NN"+"_".join([str(m) for m in modelparams])+".pmml","w") as pmml_file:
        pmml = KerasToPmml(model)
        pmml.export(pmml_file)

import tensorflow as tf
from nyoka import KerasToPmml
n_cols_in = 218
modelparams = [436,]
# get training data here; I'll toss in a CSV version.
import pandas as pd
labelcol = 'MxWillReAdmit'
trainingdata = pd.read_csv('trainingfile.csv')
x_training = np.array(trainingdata.drop(labelcol,axis='columns'))
y_training = np.array(trainingdata[labelcol])

def create_and_train(x_training,y_training,n_cols_in,modelparams,epochs=10):
    layers = [tf.keras.layers.Dense(n_cols_in,activation="relu"),
    tf.keras.layers.Dropout(.5)]
    for param in modelparams:
        layers.extend([tf.keras.layers.Dense(param,activation="sigmoid"),tf.keras.layers.Dropout(.5)])
    layers.append(tf.keras.layers.Dense(1,activation="sigmoid"))
    model = tf.keras.models.Sequential(layers)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    model.fit(x_training, y_training, epochs = epochs)
    with open("NN"+"_".join([str(m) for m in modelparams])+".pmml","w") as pmml_file:
        pmml = KerasToPmml(model)
        pmml.export(pmml_file)
create_and_train(x_training,y_training,n_cols_in,modelparams)

with open("NN"+"_".join([str(m) for m in modelparams])+".pmml","w") as pmml_file:
        pmml = KerasToPmml(model)
        pmml.export(pmml_file)

print(model.input_shape())