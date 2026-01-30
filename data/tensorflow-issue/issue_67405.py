from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, TimeDistributed, Permute
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.keras.layers import Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from keras import regularizers
from tensorflow.random import uniform
import tensorflow as tf
from tensorflow.keras.layers import Reshape
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def create_encoder_decoder_model(n_a, n_s, Tx, Ty, xFeatures, yFeatures):
   # Encoder
   encoder_inputs = Input(shape=(Tx,xFeatures))
   s1 = Input(shape=(n_a,), name='s1')#hidden state
   c1 = Input(shape=(n_a,), name='c1')#cell state
   s2 = Input(shape=(n_a,), name='s2')
   c2 = Input(shape=(n_a,), name='c2')
   s3 = Input(shape=(n_a,), name='s3')
   c3 = Input(shape=(n_a,), name='c3')
   s4 = Input(shape=(n_a,), name='s4')
   c4 = Input(shape=(n_a,), name='c4')

   encoder_lstm1,hiddenState1,cellState1 = LSTM(n_a, return_sequences=True,return_state=True)(encoder_inputs,initial_state=[s1,c1])
   encoder_lstm2,hiddenState2,cellState2 = LSTM(n_a, return_state=True)(encoder_lstm1,initial_state=[s2,c2])
   # Repeat vector to feed into decoder
   repeat_vector = RepeatVector(Ty)(encoder_lstm2)
   # Decoder
   decoder_lstm1,hiddenState3,cellState3  = LSTM(n_s, return_sequences=True,return_state=True)(repeat_vector,initial_state=[s3,c3])
   decoder_lstm2,hiddenState4,cellState4  = LSTM(n_s, return_sequences=True,return_state=True)(decoder_lstm1,initial_state=[s4,c4])
   decoder_outputs = TimeDistributed(Dense(yFeatures,activation='relu', kernel_initializer=glorot_uniform(
       seed=0), kernel_regularizer=regularizers.l2(0.01)))(decoder_lstm2)

   model = Model(inputs=[encoder_inputs, s1, c1, s2, c2, s3, c3, s4, c4], outputs=[decoder_outputs, hiddenState1, cellState1, hiddenState2, cellState2, hiddenState3, cellState3, hiddenState4, cellState4])
   return model
epochs=10
n_a=64
n_s=64
n_past=40
n_future=20
xFeatures=11
yFeatures=6
batch_size=32
X_lstm_train=np.random.random((1000,40,11))
X_lstm_val=np.random.random((100,40,11))
X_lstm_test=np.random.random((100,40,11))
Y_train = np.random.random((1000,20,6))
Y_test = np.random.random((100,20,6))
Y_val = np.random.random((1000,20,6))

s0 = np.zeros((X_lstm_train.shape[0], n_s))
c0 = np.zeros((X_lstm_train.shape[0], n_s))
s_val = np.zeros((X_lstm_val.shape[0], n_s))
c_val = np.zeros((X_lstm_val.shape[0], n_s))
s_test = np.zeros((X_lstm_test.shape[0], n_s))
c_test = np.zeros((X_lstm_test.shape[0], n_s))

model = create_encoder_decoder_model(n_a, n_s, n_past, n_future, xFeatures,yFeatures)
print(model.summary())

reduce_lr = tf.keras.callbacks.LearningRateScheduler(
    lambda x: 1e-3 * 0.90 ** x)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
#model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='model_checkpoint.h5', monitor='val_accuracy', save_best_only=True)

# model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber(),loss_weights=[1.0]+[0.0]*8)
history = model.fit([X_lstm_train, s0, c0, s0, c0, s0, c0, s0, c0],
                    Y_train,                       # Target data: Decoder outputs for training
                    epochs=epochs,
                    batch_size=batch_size,
                    # Validation data
                    validation_data=(
                        [X_lstm_val, s_val, c_val, s_val, c_val, s_val, c_val, s_val, c_val], Y_val),
                    shuffle=False  ,                                  # No shuffling
                    callbacks=[early_stopping,reduce_lr]
                )

prediction = np.array(model.predict([X_lstm_test, s_test, c_test, s_test, c_test, s_test, c_test, s_test, c_test])[0])
print(prediction.shape)