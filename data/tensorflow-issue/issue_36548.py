from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import os
import time
import tflite_runtime.interpreter as tflite
import collections
import operator

"""Functions to work with classification models."""




Class = collections.namedtuple('Class', ['id', 'score'])




def input_tensor(interpreter):
  """Returns input tensor view as numpy array of shape (height, width, 3)."""
  tensor_index = interpreter.get_input_details()[0]['index']
  return interpreter.tensor(tensor_index)()[0]

def output_tensor(interpreter):
  """Returns dequantized output tensor."""
  output_details = interpreter.get_output_details()[0] 
  output_data = np.squeeze(interpreter.tensor(output_details['index'])()) #Remove single-dimensional entries from the shape of an array.
  scale, zero_point = output_details['quantization']
  return scale * (output_data - zero_point)


def set_input(interpreter, data):
  """Copies data to input tensor."""
  input_tensor(interpreter)[:] = data
  return data   


def get_output(interpreter, top_k=1, score_threshold=0.0):
  """Returns no more than top_k classes with score >= score_threshold."""
  scores = output_tensor(interpreter)
  classes = [
      Class(i, scores[i])
      for i in np.argpartition(scores, -top_k)[-top_k:]
      if scores[i] >= score_threshold
  ]
  return sorted(classes, key=operator.itemgetter(1), reverse=True)

#load the dataset


Modelli_Prova01_Nom01_Acc1L = loadtxt(r'/home/utente/Scrivania/csvtesi/Modelli_Prova01_Nom01_Acc1L.csv',delimiter=',')
Modelli_Prova02_Nom01_Acc1L = loadtxt(r'/home/utente/Scrivania/csvtesi/Modelli_Prova02_Nom01_Acc1L.csv',delimiter=',')
Modelli_Prova03_Nom01_Acc1L = loadtxt(r'/home/utente/Scrivania/csvtesi/Modelli_Prova03_Nom01_Acc1L.csv',delimiter=',')
Modelli_Prova04_Nom01_Acc1L = loadtxt(r'/home/utente/Scrivania/csvtesi/Modelli_Prova04_Nom01_Acc1L.csv',delimiter=',')
Modelli_Prova05_Nom01_Acc1L = loadtxt(r'/home/utente/Scrivania/csvtesi/Modelli_Prova05_Nom01_Acc1L.csv',delimiter=',')

time_start = time.perf_counter()



#split x and y data (train and test)

Acc1L01_train,Acc1L01_test = train_test_split(Modelli_Prova01_Nom01_Acc1L ,test_size=0.015,random_state=42)
Acc1L02_train,Acc1L02_test = train_test_split(Modelli_Prova02_Nom01_Acc1L,test_size=0.3,random_state=42)
Acc1L03_train,Acc1L03_test = train_test_split(Modelli_Prova03_Nom01_Acc1L,test_size=0.3,random_state=42)
Acc1L04_train,Acc1L04_test = train_test_split(Modelli_Prova04_Nom01_Acc1L,test_size=0.3,random_state=42)
Acc1L05_train,Acc1L05_test = train_test_split(Modelli_Prova05_Nom01_Acc1L,test_size=0.15,random_state=42)
Y1_train= np.zeros([len(Acc1L01_train)+len(Acc1L05_train),1]) 
Y2_train= np.ones([len(Acc1L02_train),1]) 
Y3_train= np.ones([len(Acc1L03_train),1]) +1
Y4_train= np.ones([len(Acc1L04_train),1]) +2
Y1_test= np.zeros([len(Acc1L01_test)+len(Acc1L05_test),1]) 
Y2_test= np.ones([len(Acc1L02_test),1])  
Y3_test= np.ones([len(Acc1L03_test),1]) +1
Y4_test= np.ones([len(Acc1L04_test),1]) +2

xAcc1L_train = np.concatenate((Acc1L01_train,Acc1L05_train,Acc1L02_train,Acc1L03_train,Acc1L04_train),axis=0)
xAcc1L_train=MinMaxScaler([0,255]).fit_transform(xAcc1L_train)
#xAcc1L_train=StandardScaler().fit_transform(xAcc1L_train)
#xAcc1L_train=Normalizer().fit_transform(xAcc1L_train)
#xAcc1L_train=np.transpose(xAcc1L_train)
yAcc1L_train = np.concatenate((Y1_train,Y2_train,Y3_train,Y4_train),axis=0)
xAcc1L_test = np.concatenate((Acc1L01_test,Acc1L05_test,Acc1L02_test,Acc1L03_test,Acc1L04_test),axis=0)
xAcc1L_test=Normalizer().fit_transform(xAcc1L_test)
#xAcc1L_test=MinMaxScaler([0,255]).fit_transform(xAcc1L_test)
#xAcc1L_test=StandardScaler().fit_transform(xAcc1L_test)
#xAcc1L_test=np.transpose(xAcc1L_test)
yAcc1L_test = np.concatenate((Y1_test,Y2_test,Y3_test,Y4_test),axis=0)
#1 hot encode y
one_hot_labelsAcc1L =to_categorical(yAcc1L_train, num_classes=4)
one_hot_labelsAcc1L_test = to_categorical(yAcc1L_test, num_classes=4)
#fit the model
model = Sequential()
model.add(Dense(300, activation='relu', input_dim=30))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
es1 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
es2 = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=100)

history=model.fit(xAcc1L_train, one_hot_labelsAcc1L,validation_data=(xAcc1L_test,one_hot_labelsAcc1L_test),epochs=500, batch_size=30, verbose=1, callbacks=[es1,es2])
#history=model.fit(tf.cast(xAcc1L_train, tf.float32), one_hot_labelsAcc1L,validation_data=(tf.cast(xAcc1L_test, tf.float32),one_hot_labelsAcc1L_test),epochs=500, batch_size=30, verbose=1, callbacks=[es1,es2])
time_elapsed = (time.perf_counter() - time_start)
print ("%5.1f secs " % (time_elapsed))

start=time.monotonic()
_, accuracy = model.evaluate(xAcc1L_test, one_hot_labelsAcc1L_test, batch_size=30, verbose=1)
#_, accuracy = model.evaluate(tf.cast(xAcc1L_test, tf.float32), one_hot_labelsAcc1L_test, batch_size=30, verbose=1)
print(accuracy)
inference_time = time.monotonic() - start
print('%.1fms ' % (inference_time * 1000))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()  
#predicted labels
predictions = model.predict(xAcc1L_test)
y_pred = (predictions > 0.5)
matrix = confusion_matrix(one_hot_labelsAcc1L_test.argmax(axis=1), y_pred.argmax(axis=1))
print('confusion matrix = \n',matrix)
print("Accuracy:",accuracy_score(one_hot_labelsAcc1L_test.argmax(axis=1), y_pred.argmax(axis=1)))



mod01=model.save('/home/utente/Scrivania/csvtesi/rete_Nom01.h5')

#convert the model


#representative dataset       
train_ds = tf.data.Dataset.from_tensor_slices(
    (tf.cast(xAcc1L_train, tf.float32))).batch(1)
print(train_ds)

def representative_dataset_gen():
    for input_value in train_ds: 
        yield [input_value]

print(model.layers[0].input_shape)

#integer post-training quantization

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('/home/utente/Scrivania/csvtesi/rete_Nom01.h5') #all operations mapped on edge tpu
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
print(converter.representative_dataset)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()
open('/home/utente/Scrivania/csvtesi/rete_Nom01_quant.tflite', "wb").write(tflite_quant_model)



#compiler compila il modello quantizzato tflite per edge tpu
os.system("edgetpu_compiler \'/home/utente/Scrivania/csvtesi/rete_Nom01_quant.tflite'")


#interpret the model

interpreter = tf.lite.Interpreter('/home/utente/Scrivania/csvtesi/rete_Nom01_quant_edgetpu.tflite',experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])


interpreter.allocate_tensors()
idt=print(interpreter.get_input_details())
odt=print(interpreter.get_output_details())



for j in range(5):
  start = time.monotonic()
  o_test=np.arange(len(xAcc1L_test[:,0]))
  o_test=o_test[:,np.newaxis]
  for i in range (len(xAcc1L_test[:,0])):
    input=set_input(interpreter, xAcc1L_test[i,:])
    #print("inference input    %s" % input)
    interpreter.invoke()
    classes = get_output(interpreter,4)
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])#/255 con edgetpu
    #print("inference output    %s" % output)
    #print("inference classes      %s" % classes)
    a=np.array([one_hot_labelsAcc1L_test[i,:].argmax(axis=0)])
    b=np.array(output.argmax(axis=1))
    o_test[i]=b
    #if a==b:
      #print('good classification')
    #else: 
      #print('bad classification')
  inference_time = time.monotonic() - start
  print('%.1fms ' % (inference_time * 1000))
  #print(o_test)
  print("Accuracy:",accuracy_score(yAcc1L_test,o_test))