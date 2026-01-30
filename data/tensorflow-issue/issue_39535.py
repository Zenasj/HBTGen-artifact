from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow import math, dtypes
from tensorflow import float32 as f32 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import random
import numpy as np # linear algebra
import gc

rseed=10
np.random.seed(rseed)
random.seed(rseed)
tf.compat.v1.set_random_seed(rseed)
    
def MMSE( preds,targets, mask_value=0.0):
    tf.print('\npred',preds)
    tf.print('target',targets)
    mask = dtypes.cast(tf.not_equal(targets,0),f32) 
    num_rating = math.reduce_sum(mask) #count ratings
    loss = math.reduce_sum(math.square(mask*(preds - targets))) / num_rating 
    return loss


input_dim = Input(shape = (3, ))
model = Sequential()
model.add(Dense(3,input_dim=3))
model.add(Dense(3))
model.compile(optimizer = Adam(lr=0.01),loss=[MMSE]) 
            
data  = tf.math.round(tf.random.normal(shape=[5,3]))
history = model.fit(data,data, epochs = 1, batch_size = 5,verbose=0, shuffle=False) 

del input_dim,model,data,history
tf.compat.v1.reset_default_graph()
tf.keras.backend.clear_session()
gc.collect()