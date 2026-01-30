import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf 
from tensorflow.keras.layers import (Conv2D, LSTM, Dense, Dropout,
                                     Softmax, BatchNormalization, TimeDistributed)
from scipy.stats import mode
import gc

print(f'Using TensorFlow {tf.__version__}, GPU available? : {tf.test.is_gpu_available()}')

n_data = 5000
height = 9
width = 11
n_timepoints = 100
chan_dim = 1

train_x = np.random.rand(n_data, height, width, n_timepoints, chan_dim)
train_y = np.random.randint(low=0, high=3, size=n_data)
train_y = tf.keras.utils.to_categorical(train_y)

test_x = np.random.rand(n_data, height, width, n_timepoints, chan_dim)
test_y = np.random.randint(low=0, high=3, size=n_data)
test_y = tf.keras.utils.to_categorical(test_y)
                           
print(f'train_x shape: {train_x.shape}')
print(f'train_y shape: {train_y.shape}')
print(f'test_x shape: {test_x.shape}')
print(f'test_y shape: {test_y.shape}')

class mymodel(tf.keras.Model):
    
    def __init__(self, n_filters, n_fc, n_output, n_batch, n_nodes, dropout):
        super(mymodel, self).__init__(name='RCNN')
        
        # Set model properties as instance attributes
        self.n_filters = n_filters
        self.n_fc = n_fc
        self.n_output = n_output
        self.N_BATCH = n_batch
        self.N_NODES = n_nodes
        self.DROPOUT = dropout
        self.out_activation = "sigmoid" if n_output == 2 else "softmax"
        
        self.conv1 = Conv2D(filters=n_filters, strides=1,padding='same', 
                            activation='tanh', kernel_size=3)
        
        self.conv2 = Conv2D(filters=n_filters*2, strides=1,padding='same', 
                            activation='tanh', kernel_size=3)
        
        self.conv3 = Conv2D(filters=n_filters*4, strides=1,padding='same', 
                            activation='tanh', kernel_size=3)
            
        self.dense1 = Dense(n_fc)
        self.dropout1 = Dropout(dropout) 
        
        self.lstm1 = LSTM(self.N_NODES, recurrent_initializer='orthogonal', return_sequences=1)
        self.lstm2 = LSTM(self.N_NODES, recurrent_initializer='orthogonal', return_sequences=1)
        
        self.fc2         = TimeDistributed(Dense(n_fc))
        self.fc2_dropout = TimeDistributed(Dropout(dropout))
        self.outputlayer = TimeDistributed(Dense(n_output, activation=self.out_activation))
        
    def call(self, inputs, training):
        
        _, height, width, n_timesteps, n_input = inputs.shape
        inputs = tf.reshape(inputs, [self.N_BATCH*n_timesteps, height, width, 1])
        
        conv1_ = self.conv1(inputs)
        conv2_ = self.conv2(conv1_)
        conv3_ = self.conv3(conv2_)
        
        flattened_ = tf.reshape(conv3_, [-1, conv3_.shape[1]*conv3_.shape[2]*conv3_.shape[3]])
        dense1_   = self.dense1(flattened_)
        dropout1_ = self.dropout1(dense1_, training=training)
        
        lstm_in_ = tf.reshape(dropout1_, [-1, n_timesteps, self.n_fc])
        lstm1_   = self.lstm1(lstm_in_)
        lstm2_   = self.lstm2(lstm1_)
        
        fc2_         = self.fc2(lstm2_)
        fc2_dropout_ = self.fc2_dropout(fc2_, training=training)
        output_      = self.outputlayer(fc2_dropout_)
        
        return output_

optimizer = tf.keras.optimizers.Adam(1e-4)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

train_ds = tf.data.Dataset.from_tensor_slices((train_x,
                                               train_y)).batch(256, drop_remainder=True)
test_ds = tf.data.Dataset.from_tensor_slices((test_x,
                                              test_y)).batch(256, drop_remainder=True)

my_model = mymodel(n_filters=32, n_fc=256, n_batch=256, dropout=0.5,
                   n_output=train_y.shape[1], n_nodes=256)

def train_one_step(model, optimizer, x_true, y_true, training):
    with tf.GradientTape() as tape:
        y_pred = model(x_true, training)
        y_true_expanded = np.repeat(y_true, n_timepoints, axis=0).reshape((y_pred.shape[0], n_timepoints, -1))
        loss_ = loss_fn(y_true_expanded, y_pred)
        
    gradients = tape.gradient(loss_, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss_

def test_one_step(model, optimizer, x_true, y_true, training):
    y_pred = model(x_true, training)
    y_true_expanded = np.repeat(y_true, n_timepoints, axis=0).reshape((y_pred.shape[0], n_timepoints, -1))
    loss_ = loss_fn(y_true_expanded, y_pred)
    return loss_

def train(model, optimizer, train_ds, test_ds):
    
    for x_true, y_true in train_ds:
        train_loss = train_one_step(model, optimizer, x_true, y_true, training=True) 
        
    for x_true, y_true in test_ds:
        val_loss = test_one_step(model, optimizer, x_true, y_true, training=False)
            
    return (train_loss, val_loss)

for i in range(200):
    gc.collect()
    print(f'gc objects: {len(gc.get_objects())}')
    
    train_loss, val_loss = train(my_model, optimizer, train_ds, test_ds)
    tf.print(f'[TRAIN]: End (epoch {i}): loss', train_loss)
    tf.print(f'[TEST]:  End (epoch {i}): loss', val_loss)