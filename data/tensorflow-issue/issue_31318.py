import random
from tensorflow.keras import models
from tensorflow.keras import optimizers

lstm_input = Input(shape=(30, 5,))
steady_input = Input(shape=(3,),
                    name='steady_float') #тут None в shape не нужен?
dest_input = Input(shape=(1,), name='steady_dest')
ns_input = Input(shape=(1,))

x1 = layers.Bidirectional(layers.LSTM(512, activation='relu', return_sequences=True))(lstm_input)
x1 = layers.Bidirectional(layers.LSTM(256, activation='relu'))(x1)
x1 = Model(inputs=lstm_input, outputs=x1)

x2 = layers.Dense(512, activation="relu")(steady_input)
x2 = Model(inputs=steady_input, outputs=x2)

x3 = layers.Embedding(12, 3)(dest_input)
x3 = layers.Flatten()(x3)
x3 = layers.Dense(512, activation="relu")(x3)
x3 = Model(inputs=dest_input, outputs=x3)

x = layers.concatenate([x1.output, x2.output, x3.output])
x = layers.Dense(128, activation='relu')(x)

y1_output_tensor = layers.Dense(5, name='y1')(x)
y2_output_tensor = layers.Dense(5, name='y2')(x)
model = Model(inputs=[x1.input, x2.input, x3.input],
                         outputs=[y1_output_tensor, y2_output_tensor])

ep_n = 200
learning_rate = 0.001
decay_rate = learning_rate / ep_n
momentum = 0.7
model.compile(optimizer=RMSprop(lr=learning_rate, momentum=momentum, decay=decay_rate), loss=['mae', 'mae'])

#train_gen and test_get - simple generatora with shuffle
batch_size = 128
history = model.fit_generator(train_gen,
                              steps_per_epoch=1000,
                              epochs=ep_n,
                              validation_data=test_gen,
                              validation_steps=X_test.shape[0]//batch_size)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,  log_device_placement=True))
KB.set_session(sess)

ep_n = 200
learning_rate = 0.001
decay_rate = learning_rate / ep_n
momentum = 0.7
model.compile(optimizer=RMSprop(lr=learning_rate, momentum=momentum, decay=decay_rate), loss=['mae', 'mae'])

model.compile(optimizer=RMSprop(), loss=['mae', 'mae'])
KB.get_value(model.optimizer.lr)

ep_n = 200
learning_rate = 0.001
decay_rate = learning_rate / ep_n
model.compile(optimizer=RMSprop(lr=learning_rate, decay=decay_rate), loss=['mae', 'mae'])

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,  log_device_placement=True))
KB.set_session(sess)

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as KB
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

inputA = Input(shape=(1,))
inputB = Input(shape=(128,3,))
 
x = layers.Embedding(1000, 3)(inputA)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = Model(inputs=inputA, outputs=x)
 
y = layers.Bidirectional(layers.LSTM(128, activation='relu', return_sequences=True))(inputB)
y = layers.Bidirectional(layers.LSTM(64, activation='relu'))(y)
y = Model(inputs=inputB, outputs=y)

combined = layers.concatenate([x.output, y.output])
 
z = layers.Dense(2, activation="relu")(combined)
z = layers.Dense(1, activation="linear")(z)

t = layers.Dense(2, activation="relu")(combined)
t = layers.Dense(1, activation="linear")(t)
 
model = Model(inputs=[x.input, y.input], outputs=[z, t])
model.compile(RMSprop(lr=0.001, decay=0.05, momentum=0.7), ['mse', 'mse'])
#model.compile(RMSprop(lr=0.001, decay=0.05, momentum=0.0), ['mse', 'mse']) OK

input_array_a = np.random.randint(1000, size=(500000, 1))
input_array_b = np.random.randint(32, size=(500000, 128, 3))
output_array1 = np.random.randint(9, size=(500000, 1))
output_array2 = np.random.randint(9, size=(500000, 1))

def generator_shuffle(x_a, x_b, y1, y2, batch_size=128):
    max_index = len(x_a) - 1
    while 1:
        rows = np.random.randint(0, max_index, batch_size)
        yield [x_a[rows], x_b[rows]], [y1[rows], y2[rows]]

tr_gen = generator_shuffle(input_array_a,
                           input_array_b,
                           output_array1, output_array2)

history = model.fit_generator(tr_gen, steps_per_epoch=3, epochs=10)