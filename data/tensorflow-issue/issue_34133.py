from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

input_layer=Input(shape=(X.shape[1],))
model=Embedding(input_dim=len(vocab)+1,output_dim=32,input_length=X.shape[1])(input_layer)
model = Bidirectional(LSTM(units = 50, return_sequences=True, recurrent_dropout=0.2))(model)
output_layer= Dense(3, activation="softmax")(model)

model = Model(input_layer,output_layer)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
model.summary()

import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'
import tensorflow as tf
import numpy as np
from tensorflow.lite.experimental.examples.lstm.rnn import bidirectional_dynamic_rnn

def build_LSTM_layer(num_layers):
    lstm_layers=[]
    for i in range(num_layers):
        lstm_layers.append(tf.lite.experimental.nn.TFLiteLSTMCell(num_units=50,name='rnn{}'.format(i),forget_bias=1.0))
    final_lstm_layer=tf.keras.layers.StackedRNNCells(lstm_layers)
    return final_lstm_layer
def build_bidirectional(inputs,num_layers,use_dynamic_rnn=True):
    lstm_inputs=transposed_inp=tf.transpose(inputs,[1,0,2])
    outputs,output_states=bidirectional_dynamic_rnn(build_LSTM_layer(num_layers),build_LSTM_layer(num_layers),lstm_inputs,dtype="float",time_major=True)
    fw_lstm_output,bw_lstm_output=outputs
    final_out=tf.concat([fw_lstm_output,bw_lstm_output],axis=2)
    
    final_out=tf.unstack(final_out,axis=0)
   
    resultant_out=final_out[-1]
    
    return resultant_out

tf.reset_default_graph()

model_tf = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(X.shape[1],), name='input'),  
  tf.keras.layers.Embedding(input_dim=len(vocab)+1,output_dim=32,input_length=X.shape[1]),
  tf.keras.layers.Lambda(build_bidirectional, arguments={'num_layers' : 2, 'use_dynamic_rnn': True}),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax, name='output')
])
model_tf.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model_tf.summary()