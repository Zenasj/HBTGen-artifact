import numpy as np
import tensorflow as tf

def weight(d):
  f = model_weight.predict(d,steps = 1)
  return (f)/(1-f)

myinputs = Input(shape=(1,), dtype = tf.float32)
x = Dense(128, activation='relu')(myinputs)
x2 = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x2)
model = Model(inputs=myinputs, outputs=predictions)
model.summary()

def my_loss_wrapper(inputs,val=0):
  x  = inputs
  theta = 0. #starting value
  #theta0 = tf.constant(val, dtype= tf.float32)#target value

  #creating tensor (filled with 1s) with same shape as inputs; multiply by val
  theta0_stack = K.ones_like(x, dtype=tf.float32)*val 

  #combining and reshaping into correct format:
  data = K.stack((x, theta0_stack), axis=-1) 
  data = K.squeeze(data, axis = 1)
  #slice data to 500 entries to match batch_size
  data = K.gather(data, np.arange(500))
  print(data.shape)

  w = weight(data)

  def my_loss(y_true,y_pred):
    t_loss = K.mean(y_true*(y_true - y_pred)**2+(w)**2*(1.-y_true)*(y_true - y_pred)**2)
    return t_loss

  return my_loss

model.compile(optimizer='adam', loss=my_loss_wrapper(myinputs,theta),metrics=['accuracy'])
model.fit(np.array(X_train), y_train, epochs=1, batch_size=500,validation_data=(np.array(X_test), y_test),verbose=1)