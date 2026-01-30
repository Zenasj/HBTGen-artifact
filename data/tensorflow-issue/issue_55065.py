import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

def My_LSTM(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Masking(mask_value= -27/255 , input_shape=(None, 600)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(16,activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(4,activation='softmax'))

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer= Adam(learning_rate = 1e-3, decay = 1e-6),metrics = ['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-9)
    stop = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

    
    model.fit(X_train, y_train, epochs=40, batch_size= 128, validation_data=(X_test, y_test), callbacks=[reduce_lr, stop], verbose =1)
    
    
    return model

def Create_tflite_Model(ModelName, savePath):
    
    keras_model = tf.keras.models.load_model(ModelName, compile = False)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter=True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    tflite_model_dir = pathlib.Path(savePath)
    tflite_model_file = tflite_model_dir/"My_model.tflite"
    tflite_model_file.write_bytes(tflite_model)
    
    return tflite_model

tf.lite.Interpreter

Super_Interpreter = tf.lite.Interpreter(model_path='My_model.tflite')
Super_Interpreter.get_signature_list()
Super = Super_Interpreter.get_signature_runner('serving_default')
Result = Super(masking_input = "File to Classify")
object = np.argmax(Result.get('dense_1'))
print(Classes[object])

import numpy as np
import os
import time
import tflite_runtime.interpreter as tflite

Class = ['Vehicle_1', 'Vehicle_2', 'Vechile_3', 'Vechile_4']

Super_Interpreter = tflite.Interpreter(model_path='My_model.tflite')
Super_Interpreter.get_signature_list()
Super = Super_Interpreter.get_signature_runner('serving_default')

def Preprocess(txtfile):
    
    Sample = [] 
    Sample_padded = [] 
    frames = {} 

    array_from_file = np.loadtxt(txtfile, delimiter=',', dtype=float)
    array_from_file = (array_from_file*100).astype(int)
    array_from_file_list = array_from_file.tolist()
        
    for item in array_from_file_list: 
        if item[1] in frames:
            frames[item[1]].append([item[0], item[2]])
        else:
            frames[item[1]] = [[item[0], item[2]]]
    
    for key in frames:
        Sample.append(frames[key]) 
    
    for points in Sample: 
        Sample_padded.append(np.pad(points,[(0,300 - len(points)), (0,0)], 'constant', constant_values=(-27, -27)))
    
    Detect = np.array(Sample_padded).reshape(1,-1,600)/255
    Detect = Detect.astype(np.float32)

    return Detect

for i in os.listdir():
      if i.endswith('.txt'):
            start_time = time.time()
            Sample_file = Preprocess(i)
            Result = Super(masking_input = Sample_file)
            Vehicle = np.argmax(Result.get('dense_1'))
            print(Class[Vehicle])
            print("--- %s seconds ---" % (time.time() - start_time))

tf.lite.Interpreter

tflite_runtime