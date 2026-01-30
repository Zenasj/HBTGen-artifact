import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

# get xception
inshape = (256,256,3)
ptdnn = tf.keras.applications.Xception(
	weights='imagenet',
  	include_top=False,
  	input_shape=inshape)
# build classifier
classifier = tf.keras.models.Sequential(name='Classifier')
  # The classifier input is dense, or fully connected
classifier.add(tf.keras.layers.Dense(256, activation='relu',
  input_shape=feature_shape[1:]))
  # The flatten layer reduces the dimensions of the output
classifier.add(tf.keras.layers.Flatten())
  # Dropout layer prevents overfitting
classifier.add(tf.keras.layers.Dropout(0.5))
  # Output layer is a single neuron sigmoid
classifier.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#Assemble composite model
model = tf.keras.models.Sequential()
  #First get the features from ptdnn
model.add(ptdnn)
  #Then classify
model.add(classifier)  
  #feature_detector should not be trainable if fine_tuning_layers==0
ptdnn.trainable=False

  # Display layer information for reference
print('[ASSEMBLE] Full Model Architecture:')
model.summary()

  # classifier options including metrics and loss function
classifier.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-4),
                loss='binary_crossentropy',
                metrics=['acc'])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                loss='binary_crossentropy',
                metrics=['acc'])

# try to fit on random data 
batch_size=1
model.train_on_batch(np.random.rand(batch_size,256,256,3),np.random.rand(batch_size,))

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    """ F1 metric.
    The F1 metric is the harmonic mean of precision and recall
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))