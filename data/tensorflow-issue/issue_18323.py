from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

-traceback
with open(str(cnt) + ".txt", 'w') as f:        
        print(tf.get_default_graph().as_graph_def(), file=f)

-traceback
file1 = open("1.txt", 'r')
file2 = open("3.txt", 'r')
Dict1 = file1.readlines()
Dict2 = file2.readlines()
DF = [ x for x in Dict1 if x not in Dict2 ]
print(DF)

-traceback
import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
np.random.seed(12)
import random as rn
rn.seed(123)
import tensorflow as tf
tf.set_random_seed(1234)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)

from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense

lag = 5

train = values[:(450 - lag - 1), :]
test = values[(450 - lag - 1):, :]

num_parameters = 5
feature = num_parameters - 1

train_X, train_y = train[:, 0:num_parameters * lag], train[:, feature - num_parameters]
train_X = train_X.reshape((train_X.shape[0], lag, num_parameters))

test_X, test_y = test[:, 0:num_parameters * lag], test[:, feature - num_parameters]
test_X = test_X.reshape((test_X.shape[0], lag, num_parameters))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

tbCallBack = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=32,
write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, 
embeddings_layer_names=None, embeddings_metadata=None)

#===============================================

for counter in range(1, 11):    
            
    model = Sequential()
    model.add(LSTM(5, input_shape=(lag, num_parameters), return_sequences=False))
    model.add(Dense(1, activation = "linear"))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    history = model.fit(train_X, train_y, epochs=5, batch_size=1, validation_data=(test_X, test_y), 
                    verbose = 2, shuffle = False, callbacks=[tbCallBack])    
          
    with open(str(counter) + ".txt", 'w') as f:        
        print(tf.get_default_graph().as_graph_def(), file=f)
        
    print("Counter=", counter)    
    del model
    K.clear_session() 
    tf.reset_default_graph()
    np.random.seed(12)    
    rn.seed(123)    
    tf.set_random_seed(1234)