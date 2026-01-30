import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np

# use keras
from keras.layers import Dense, Input
from keras.models import Model

# use tf.keras
# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras.models import Model

# define input
noise = Input(shape=(10,))
x = Input(shape=(100,))

# define generator and discriminator
gen = Dense(100)
dis = Dense(1)

y = dis(x)
dis_model = Model(x, y)
dis_model.compile(optimizer='rmsprop', loss='mse')
dis_model.summary()

z = dis_model(gen(noise))
dis_model.trainable = False
combined_model = Model(noise, z)
combined_model.compile(optimizer='rmsprop', loss='mse')
combined_model.summary()

for i in range(3):
    dis_model.train_on_batch(x=np.random.rand(10, 100),
                             y=np.random.rand(10, 1))
    combined_model.train_on_batch(x=np.random.rand(10, 10),
                             y=np.random.rand(10, 1))

model._collected_trainable_weights

model.compile

dis_model.trainable = False

dis_model

dis_model

dis_model._collected_trainable_weights

dis_model.trainable_weights

dis_model.train_on_batch

dis_model._collected_trainable_weights

dis_model._collected_trainable_weights

dis_model.trainable_weights

combined_model.compile

combined_model._collected_trainable_weights

dis_model

tf.keras