import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.random.set_seed(1)
BATCH_SIZE = 32

#Import mnist dataset as numpy arrays
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()#Import
x_train = x_train / 255.0 #normalizing
y_train = y_train.astype(dtype='float32')
x_train = x_train.astype(dtype='float32')

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))#Reshaping the 2D picture

##############################################################################################
#THIS BLOCK CREATES A DATASET FROM THE NUMPY ARRAYS. IT WILL BE USED FOR THE CASE OF TF.DATA DATASET INPUTS
tfdata_dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
tfdata_dataset_train = tfdata_dataset_train.batch(BATCH_SIZE).repeat()
##############################################################################################

#Create model
keras_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2, seed=1),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

metrics = [tf.metrics.Accuracy()]
#metrics = ['accuracy']

#Compile the model
keras_model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.keras.losses.sparse_categorical_crossentropy,
                    metrics=metrics)


#Train with tf.data datasets
keras_training_history = keras_model.fit(tfdata_dataset_train,
                epochs=1,
                steps_per_epoch=60000//BATCH_SIZE
                )
########################

"""Bug."""
# import keras
import numpy as np
import tensorflow.keras as keras

X = np.empty([10, 224, 224, 3])
Y = np.empty([10, 2])

MODEL = keras.applications.vgg16.VGG16(weights=None, classes=2)

MODEL.compile(optimizer=keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
MODEL.fit(X, Y, epochs=10)

MODEL.compile(optimizer=keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=[keras.metrics.get('accuracy')])
MODEL.fit(X, Y, epochs=10)