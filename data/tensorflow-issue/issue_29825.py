import numpy as np
from tensorflow import keras

def split_sequence(features, labels, window_size):
    X, y = list(), list()
    rng = len(features) - window_size
    for i in range(rng):
        last_ix = i + window_size
        # gather input and output parts of the pattern
        seq_x, seq_y = features[i:last_ix], labels[last_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.asarray(X), np.asarray(y)

import tensorflow as tf
import keras

window_size = 1024
n_steps = len(data[0]) - window_size
inputs_n = 7
outputs_n = 4
neurons = 128
learning_rate = 0.00001
activation = 'softmax'

from keras.layers import Dense, Activation, Dropout, LSTM, Dropout
from tensorflow.keras import layers

tf.keras.backend.clear_session()
tf.keras.backend.get_session().run(tf.global_variables_initializer())

model = tf.keras.Sequential()
model.add(layers.LSTM(neurons, input_shape=(window_size, inputs_n), return_sequences=True)) 
model.add(layers.LSTM(neurons))
model.add(layers.Dense(neurons, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(outputs_n, activation=activation))

opt = tf.train.RMSPropOptimizer(learning_rate)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print(model.summary())

import time
# loop by day - separate features and labels

buy_sell = 'sell'
print('Training.. LABEL:', buy_sell, 'neurons:', neurons, 'learning rate:', learning_rate, 'activation:', activation)

epochs_n = 1
epochs =  range(epochs_n)

histories = list()
day_count = 0

start_time = time.time()

# d is tuple from groupby - d[0] = date, d[1] = values
for epoch in epochs:
    for d in data : 
        # get arrays for the day
        features = np.asarray(d)[:,2:9].astype(dtype = 'float32')
        labels = np.asarray(d)[:, 9:13].astype(dtype = 'int8')
        
        X,y = split_sequence(features, labels, window_size)

        try:
            H = model.fit(X,y, batch_size = window_size)
            histories.append(H.history)
        except Exception as e:
            print('** train exception :', e)
            continue
        
    #for days
#for epoch

print('DONE..')

0.322791712104689,0.323336968375136,0.00109051254089421,6.4610961249576E-05,0.746954076850984,0.7572633552015,0.746954076850984,0,1,0,0
0.323882224645583,0.323882224645583,0,6.4610961249576E-05,0.751640112464855,0.801312089971884,0.751640112464855,0,0,0,1
0.323882224645583,0.324427480916031,0.00109051254089421,0.00928782567962655,0.792877225866917,0.817244611059044,0.792877225866917,0,0,1,0
0.323882224645583,0.324427480916031,0,0.00568576458996269,0.837863167760075,0.837863167760075,0.808809746954077,0,0,1,0
0.322791712104689,0.323336968375136,0.00109051254089421,0.000516887689996608,0.820056232427366,0.820056232427366,0.799437675726336,0,0,0,0
0.323882224645583,0.323882224645583,0,3.2305480624788E-05,0.799437675726336,0.817244611059044,0.792877225866917,0,0,0,1