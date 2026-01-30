import numpy as np

embd=Embedding(input_dim=len(vocab),output_dim=100,name="embd")
lstm1=Bidirectional(LSTM(units=100,return_sequences=True,name="lstm1"),name="bd1")
lstm2=Bidirectional(LSTM(units=100,return_sequences=True,name="lstm2"),name="bd2")
attention_layer=Attention_Model(21,200)
dense1=Dense(units=80,name="dense1",kernel_regularizer="l2")
dropout1=Dropout(0.5)
act1=Activation('sigmoid')

dense2=Dense(units=50,name="dense2",kernel_regularizer="l2")
dropout2=Dropout(0.4)
act2=Activation('sigmoid')

dense3=Dense(units=30,name="dense3",kernel_regularizer="l2")
dropout3=Dropout(0.3)
act3=Activation('sigmoid')

dense4=Dense(units=len(classes),name="dense4")
dropout4=Dropout(0.2)
output=Activation('softmax')

def forward_pass(X):
  t=embd(X)
 
  t=lstm1(t)
  
  t=lstm2(t)
  

 
  t=attention_layer(t)
  
  
  t=dense1(t)
  t=dropout1(t)
  t=act1(t)

  t=dense2(t)
  t=dropout2(t)
  t=act2(t)

  t=dense3(t)
  t=dropout3(t)
  t=act3(t)
  
  t=dense4(t)
  t=dropout4(t)
  t=output(t)

  return t

class Attention_Model():
  def __init__(self,seq_length,units):
    self.seq_length=seq_length
    self.units=units
    self.lstm=LSTM(units=units,return_sequences=True,return_state=True)
    

  def get_lstm_s(self,seq_no):
    input_lstm=tf.expand_dims(tf.reduce_sum(self.X*(self.alphas[:,:,seq_no:seq_no+1]),axis=1),axis=1)
    a,b,c=self.lstm(input_lstm)
    self.output[:,seq_no,:]=a[:,0,:]

    return b

  def __call__(self,X):
    self.X=X
    self.output=np.zeros(shape=(self.X.shape[0],self.seq_length,self.units))
    self.dense=Dense(units=self.seq_length)
    self.softmax=Softmax(axis=1)
    

    for i in range(self.seq_length+1):
      if i==0 :
        s=np.zeros(shape=(self.X.shape[0],self.units))
      else :
        s=self.get_lstm_s(i-1)
      if(i==self.seq_length):
        break 
      
      s=RepeatVector(self.X.shape[1])(s)
      concate_X=np.concatenate([self.X,s],axis=-1)
      
      self.alphas=self.softmax(self.dense(concate_X))

    return self.output

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf