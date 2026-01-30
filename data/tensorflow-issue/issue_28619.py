import random
from tensorflow.keras import layers
from tensorflow.keras import models

from sklearn.metrics import roc_auc_score
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import numpy as np
from functools import partial

def auc(weight):
    def metric(y_true, y_pred):
        score = tf.py_func(partial(roc_auc_score, sample_weight=weight), (y_true, y_pred), tf.float32)
        K.get_session().run(tf.local_variables_initializer())
        return score
    return metric

x=Input(shape=(10, ))
weights = Input(shape=(1,))
hidden = Dense(10, activation='relu')(x)
result = Dense(1, activation='sigmoid')(hidden)
model = Model(inputs=[x, weights], outputs=result)
model.compile('adam', 'binary_crossentropy', metrics=[auc(weights)])

X = np.random.rand(10000, 10)
y = np.random.randint(2, size=(10000, 1))
w = np.random.rand(10000, 1)
X_val = np.random.rand(100, 10)
y_val = np.random.randint(2, size=(100, 1))
w_val = np.random.rand(100, 1)
model.fit([X, w], y, epochs=20, sample_weight=w.flatten(), validation_data=([X_val, w_val], y_val), verbose=2)